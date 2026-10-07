// job-band: a live row above the prompt for every long job on this machine.
//
// THE PROTOCOL. A job writes `<repo>/.progress/<anything>.json`, rewritten as it
// goes (atomically: write a temp file, rename it over):
//
//   { "label": "pytest regression", "done": 812, "total": 1664, "unit": "tests",
//     "failed": 0, "state": "running" | "passed" | "failed",
//     "started_at": <epoch s>, "updated_at": <epoch s>, "detail": "..." }
//
// `updated_at` is a HEARTBEAT, not the last progress: a writer refreshes it every
// few seconds even while one long test runs, so a heartbeat that stops means the
// job died or the machine slept — which is what this band exists to say out loud.
// `tests/report_plugin.py` writes one per pytest run; anything else can join.
import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { AgentRow, Job, Snapshot } from '../types'

const snapshot = atom({ plugin: 'job-band', key: 'snapshot' } as const, { now: 0, jobs: [], agents: [] } as Snapshot)
const isHidden = atom({ plugin: 'job-band', key: 'isHidden' } as const, false)

const TICK_MS = 250 // the spinner's frame rate
const READ_EVERY = 4 // ticks between directory reads (once a second)
const STALL_S = 30 // no heartbeat for this long: say so
const KEEP_FINISHED_S = 60 // a finished job stays on the band this long
const PRUNE_FINISHED_S = 600 // ... and its file is removed after this
const KEEP_STALLED_S = 30 * 60

const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const PARTIAL = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉']

export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h) return `${h}h${String(m).padStart(2, '0')}m`
  if (m) return `${m}m${String(r).padStart(2, '0')}s`
  return `${r}s`
}

export function bar(fraction: number, width = 24): string {
  const f = Math.min(1, Math.max(0, fraction))
  const eighths = Math.round(f * width * 8)
  const full = Math.floor(eighths / 8)
  const part = PARTIAL[eighths % 8]
  return ('█'.repeat(full) + part).padEnd(width, '·')
}

/** A progress file's text -> a Job, or null when it is not one. */
export function parseJob(file: string, text: string): Job | null {
  let raw: Record<string, unknown>
  try {
    raw = JSON.parse(text)
  } catch {
    return null // caught mid-write by a writer that does not rename: next read
  }
  const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)
  const started = num(raw.started_at)
  const updated = num(raw.updated_at)
  if (started === null || updated === null) return null
  const state = raw.state === 'passed' || raw.state === 'failed' ? raw.state : 'running'
  return {
    file,
    label: typeof raw.label === 'string' ? raw.label : file.replace(/\.json$/, ''),
    done: num(raw.done) ?? 0,
    total: num(raw.total),
    unit: typeof raw.unit === 'string' ? raw.unit : '',
    failed: num(raw.failed) ?? 0,
    state,
    startedAt: started,
    updatedAt: updated,
    detail: typeof raw.detail === 'string' ? raw.detail : '',
    pid: pidOf(file),
  }
}

/** `<name>-<pid>.json` -> pid, or null. */
export function pidOf(file: string): number | null {
  const m = /-(\d+)\.json$/.exec(file)
  return m ? Number(m[1]) : null
}

/** Whether a job's file should be removed: finished long ago, or running with
 *  a process that is gone (a killed run). A LIVE process with a stale
 *  heartbeat — a machine that slept — is kept: that warning is the point. */
export function isCruft(job: Job, now: number, alive: boolean): boolean {
  if (job.state !== 'running') return now - job.updatedAt > PRUNE_FINISHED_S
  return !alive && job.pid !== null
}

export function isStalled(job: Job, now: number): boolean {
  return job.state === 'running' && now - job.updatedAt > STALL_S
}

/** Whether the band still shows a job at `now` (epoch seconds). */
export function isShown(job: Job, now: number): boolean {
  if (job.state !== 'running') return now - job.updatedAt <= KEEP_FINISHED_S
  return now - job.updatedAt <= KEEP_STALLED_S
}

/** Seconds left at the job's average rate so far, or null when unknowable. */
export function eta(job: Job, now: number): number | null {
  if (job.total === null || job.done <= 0 || job.done >= job.total) return null
  const spent = now - job.startedAt
  if (spent <= 0) return null
  return ((job.total - job.done) * spent) / job.done
}

/** The row's words, split so the renderer can colour each part. */
export function describe(job: Job, now: number) {
  const end = job.state === 'running' ? now : job.updatedAt
  const elapsed = duration(end - job.startedAt)
  const fraction = job.total ? job.done / job.total : 0
  const count = job.total
    ? `${job.done.toLocaleString('en-US')}/${job.total.toLocaleString('en-US')}${job.unit ? ' ' + job.unit : ''}`
    : `${job.done.toLocaleString('en-US')}${job.unit ? ' ' + job.unit : ''}`
  const failed = job.failed ? `${job.failed} failed` : ''
  let tail: string
  if (job.state === 'passed') tail = `done in ${elapsed}`
  else if (job.state === 'failed') tail = `finished in ${elapsed}`
  else if (isStalled(job, now))
    tail = `${elapsed} · NO HEARTBEAT for ${duration(now - job.updatedAt)} — stopped, or the machine slept?`
  else {
    const left = eta(job, now)
    tail = `${elapsed} elapsed${left === null ? '' : ` · ~${duration(left)} left`}`
  }
  return {
    pct: job.total ? `${Math.floor(fraction * 100)}%`.padStart(4) : '',
    bar: job.total ? bar(fraction) : '',
    count,
    failed,
    tail,
  }
}

// ── Agents ──────────────────────────────────────────────────────────────
//
// A subagent has no progress file and no true ETA: it decides how much work it
// does. What the band can say honestly is that it is ALIVE (its tool calls, the
// one it is on now), how long it has run, and how long that agent type usually
// takes — the median of its past runs, kept across sessions in `$.store`.
const LIVE = new Set(['pending', 'running', 'waiting'])
const KEEP_RUNS = 10

export function median(xs: number[]): number | null {
  if (xs.length === 0) return null
  const s = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2
}

/** One tool call, said in a few words: what an agent is doing right now. */
export function summarize(e: Record<string, unknown>): string {
  const tool = String(e.tool ?? '')
  const what = [e.description, e.command, e.file_path, e.pattern, e.query, e.url]
    .find(v => typeof v === 'string' && v.length > 0) as string | undefined
  const text = what ? `${tool}: ${what.replace(/\s+/g, ' ')}` : tool
  return text.length > 70 ? text.slice(0, 69) + '…' : text
}

/** `sla_s: 120` from a charter's YAML frontmatter, or null when it declares none. */
export function parseSla(charter: string): number | null {
  const head = /^---\n([\s\S]*?)\n---/.exec(charter)
  const m = head ? /^sla_s:\s*(\d+(?:\.\d+)?)\s*$/m.exec(head[1]!) : null
  return m ? Number(m[1]) : null
}

/** Over its response-time target? Floors (a late row) are never judged. */
export function overSla(a: AgentRow, now: number): boolean {
  return a.sla !== null && !a.late && (a.endedAt ?? now) - a.startedAt > a.sla
}

export function agentTail(a: AgentRow, now: number): string {
  const end = a.endedAt ?? now
  const elapsed = (a.late ? '≥' : '') + duration(end - a.startedAt)
  const tools = `${a.late ? '≥' : ''}${a.tools} tool call${a.tools === 1 ? '' : 's'}`
  const target = a.sla === null || a.late ? ''
    : overSla(a, now) ? ` · OVER its ${duration(a.sla)} target by ${duration(end - a.startedAt - a.sla)}`
    : ` · target ${duration(a.sla)}`
  if (a.endedAt !== null) return `${a.status} in ${elapsed} · ${tools}${target}`
  if (target) return `${elapsed} elapsed · ${tools}${target}`
  let usual = ''
  if (a.typical !== null && !a.late) {
    const left = a.typical - (now - a.startedAt)
    usual = left > 0
      ? ` · usually ~${duration(a.typical)} (${a.runs} run${a.runs === 1 ? '' : 's'}), ~${duration(left)} to go`
      : ` · past its usual ~${duration(a.typical)} by ${duration(-left)}`
  }
  return `${elapsed} elapsed · ${tools}${usual}`
}

async function isAlive($: EngineInterface, pid: number | null): Promise<boolean> {
  if (pid === null) return true
  try {
    return (await $.process.run(['kill', '-0', String(pid)])).exitCode === 0
  } catch {
    return true // cannot tell: keep it rather than hide a real job
  }
}

async function remove($: EngineInterface, path: string): Promise<void> {
  try {
    await $.process.run(['rm', '-f', path])
  } catch {
    // the writers prune too (manamap.progress), so a refusal here only delays it
  }
}

/** Read every progress file; remove the cruft (`all`: every non-running one). */
async function readJobs($: EngineInterface, root: string, all = false): Promise<Job[]> {
  const dir = `${root}/.progress`
  if (!(await $.fs.exists(dir))) return []
  const now = (await $.clock.now()) / 1000
  const jobs: Job[] = []
  for (const entry of await $.fs.list(dir)) {
    if (entry.kind !== 'file' || !entry.name.endsWith('.json')) continue
    let job: Job | null = null
    try {
      job = parseJob(entry.name, await $.fs.read(`${dir}/${entry.name}`))
    } catch {
      continue // renamed away between the list and the read: the next tick sees it
    }
    if (!job) continue
    const stale = job.state === 'running' && isStalled(job, now)
    const alive = stale ? await isAlive($, job.pid) : true
    if ((all && job.state !== 'running') || isCruft(job, now, alive)) {
      await remove($, `${dir}/${entry.name}`)
      continue
    }
    jobs.push(job)
  }
  return jobs.sort((a, b) => a.startedAt - b.startedAt)
}

type Seen = { startedAt: number; endedAt: number | null; status: string; tools: number; last: string; late: boolean }

/** Each agent type's target, read once a session from `.claude/agents/<type>.md`. */
const slas = new Map<string, number | null>()

async function slaOf($: EngineInterface, root: string, type: string): Promise<number | null> {
  if (!slas.has(type)) {
    let sla: number | null = null
    try {
      sla = parseSla(await $.fs.read(`${root}/.claude/agents/${type}.md`))
    } catch {
      // a built-in agent type has no charter here, and so no target
    }
    slas.set(type, sla)
  }
  return slas.get(type) ?? null
}

/** One line per finished run with a target, so a missed target is on the record
 * (`manamap pilot sla-report` reads it). Kept to the last 500 runs. */
async function logRun($: EngineInterface, root: string, row: Record<string, unknown>): Promise<void> {
  const path = `${root}/.progress/sla-log.jsonl`
  try {
    const old = (await $.fs.exists(path)) ? await $.fs.read(path) : ''
    const lines = old.split('\n').filter(l => l.trim()).slice(-499)
    await $.fs.write(path, [...lines, JSON.stringify(row)].join('\n') + '\n')
  } catch {
    // the log is a record, never a reason to break the band
  }
}

async function readAgents($: EngineInterface, root: string, seen: Map<string, Seen>, now: number, first: boolean): Promise<AgentRow[]> {
  const rows: AgentRow[] = []
  for (const a of await $.agent.list()) {
    const live = LIVE.has(a.status)
    let s = seen.get(a.id)
    if (!s) {
      if (!live) continue // ended before the band saw it start: no honest duration
      // Live on the band's FIRST read: it started before the band loaded, so its
      // start (and tool count) are unknown — shown as a floor, never recorded.
      s = { startedAt: now, endedAt: null, status: a.status, tools: 0, last: '', late: first }
      seen.set(a.id, s)
    }
    const key = `durations:${a.type}`
    const past = ((await $.store.get(key)) as number[] | undefined) ?? []
    const sla = await slaOf($, root, a.type)
    if (!live && s.endedAt === null) {
      s.endedAt = now
      if (a.status === 'completed' && !s.late) {
        await $.store.set(key, [...past, now - s.startedAt].slice(-KEEP_RUNS))
      }
      if (sla !== null && !s.late) {
        const elapsed = Math.round((now - s.startedAt) * 10) / 10
        await logRun($, root, { type: a.type, status: a.status, elapsed_s: elapsed, sla_s: sla,
          missed: elapsed > sla, at: new Date(now * 1000).toISOString(), description: a.description })
      }
    }
    s.status = a.status
    if (s.endedAt !== null && now - s.endedAt > KEEP_FINISHED_S) continue
    rows.push({
      id: a.id, type: a.type, description: a.description, status: a.status,
      startedAt: s.startedAt, endedAt: s.endedAt, tools: s.tools, last: s.last,
      typical: median(past), runs: past.length, late: s.late, sla,
    })
  }
  return rows
}

export const register: Register = on => {
  let root = ''
  let ticks = 0
  const seen = new Map<string, Seen>()

  // Every tool call inside a subagent's loop carries its id: count them, and
  // keep the latest as "what it is doing now".
  on('tool.call', ($, e, next) => {
    if (e.agentId) {
      const s = seen.get(e.agentId)
      if (s) {
        s.tools += 1
        s.last = summarize(e as unknown as Record<string, unknown>)
      }
    }
    return next(e)
  })

  on('session.start', async ($, e, next) => {
    root = await $.session.root()
    await $.command.register({ name: 'jobs', description: 'Show or hide the job band; `/jobs clear` removes every finished row' })
    await readJobs($, root, true) // a new session starts with a clean band
    $.clock.every(TICK_MS, async () => {
      const now = (await $.clock.now()) / 1000
      if (ticks++ % READ_EVERY === 0) {
        const jobs = await readJobs($, root)
        const agents = await readAgents($, root, seen, now, ticks === 1)
        await update($, snapshot, () => ({ now, jobs, agents }))
      } else {
        // a frame of the spinner only: the same rows, a new clock
        await update($, snapshot, s =>
          s.jobs.some(j => isShown(j, now)) || s.agents.length ? { ...s, now } : s)
      }
    })
    return next(e)
  })

  on('command.run', { command: 'jobs' }, async ($, e) => {
    if (/\bclear\b/.test(String((e as unknown as { args?: string }).args ?? ''))) {
      const jobs = await readJobs($, root, true)
      const now = (await $.clock.now()) / 1000
      await update($, snapshot, s => ({ ...s, now, jobs, agents: s.agents.filter(a => a.endedAt === null) }))
      return { text: `Job band cleared — ${jobs.length} running job(s) kept.` }
    }
    const hidden = await update($, isHidden, h => !h)
    return { text: hidden ? 'Job band hidden. /jobs shows it again.' : 'Job band shown.' }
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const { now, jobs, agents } = await read($, snapshot)
    const shown = jobs.filter(j => isShown(j, now))
    if (e.props.hasSurvey || (shown.length === 0 && agents.length === 0) || (await read($, isHidden)))
      return next(e)

    const { Box, Text } = $.ui.resolve(e)
    const frame = SPINNER[Math.floor((now * 1000) / TICK_MS) % SPINNER.length]

    return (
      <Box flexDirection="column">
        {shown.map(job => {
          const d = describe(job, now)
          const stalled = isStalled(job, now)
          const colour =
            job.state === 'passed' ? 'green' : job.state === 'failed' || job.failed ? 'red' : stalled ? 'yellow' : 'cyan'
          const icon = job.state === 'passed' ? '✔' : job.state === 'failed' ? '✘' : stalled ? '⏸' : frame
          return (
            <Box key={job.file}>
              <Text color={colour}>{icon} </Text>
              <Text bold>{job.label} </Text>
              {d.bar ? <Text color={colour}>{d.bar} </Text> : null}
              {d.pct ? <Text bold>{d.pct} </Text> : null}
              <Text dimColor>{d.count} </Text>
              {d.failed ? <Text color="red">{d.failed} </Text> : null}
              <Text color={stalled ? 'yellow' : undefined} dimColor={!stalled}>
                {d.tail}
              </Text>
            </Box>
          )
        })}
        {agents.map(a => {
          const done = a.endedAt !== null
          const late = overSla(a, now)
          const colour = !done ? (late ? 'yellow' : 'magenta') : a.status === 'completed' ? (late ? 'yellow' : 'green') : 'red'
          const icon = !done ? frame : a.status === 'completed' ? '✔' : '✘'
          return (
            <Box key={a.id} flexDirection="column">
              <Box>
                <Text color={colour}>{icon} </Text>
                <Text bold>{a.type} </Text>
                <Text>{a.description} </Text>
                <Text color={late ? 'yellow' : undefined} dimColor={!late}>{agentTail(a, now)}</Text>
              </Box>
              {!done && a.last ? (
                <Box>
                  <Text dimColor>{'   ↳ '}{a.last}</Text>
                </Box>
              ) : null}
            </Box>
          )
        })}
      </Box>
    )
  })
}
