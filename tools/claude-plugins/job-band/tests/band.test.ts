import { describe as group, expect, test } from 'claude-code/testing'

import { bar, describe, duration, eta, isShown, isStalled, parseJob } from '../hooks/register'

const T0 = 1_000_000

function job(over: Record<string, unknown> = {}) {
  const j = parseJob('pytest-1.json', JSON.stringify({
    label: 'pytest regression', done: 400, total: 1600, unit: 'tests', failed: 0,
    state: 'running', started_at: T0, updated_at: T0 + 600, ...over,
  }))
  if (!j) throw new Error('fixture did not parse')
  return j
}

group('job-band', () => {
  test('ETA is the average rate so far: a quarter done in 10m leaves 30m', async () => {
    expect(eta(job(), T0 + 600)).toBe(1800)
    expect(describe(job(), T0 + 600).tail).toBe('10m00s elapsed · ~30m00s left')
  })

  test('a heartbeat that stops is said out loud — the laptop-asleep case', async () => {
    const asleep = job({ updated_at: T0 + 600 })
    expect(isStalled(asleep, T0 + 600 + 29)).toBe(false)
    expect(isStalled(asleep, T0 + 600 + 6600)).toBe(true)
    expect(describe(asleep, T0 + 7200).tail).toContain('NO HEARTBEAT for 1h50m')
  })

  test('a finished job reports its own duration, stays one minute, then goes', async () => {
    const done = job({ state: 'passed', done: 1600, updated_at: T0 + 900 })
    expect(describe(done, T0 + 950).tail).toBe('done in 15m00s')
    expect(isShown(done, T0 + 900 + 60)).toBe(true)
    expect(isShown(done, T0 + 900 + 61)).toBe(false)
    expect(isStalled(done, T0 + 99_999)).toBe(false)
  })

  test('the bar fills in eighths and never overflows', async () => {
    expect(bar(0, 8)).toBe('········')
    expect(bar(0.5, 8)).toBe('████····')
    expect(bar(1 / 16, 8)).toBe('▌·······')
    expect(bar(2, 8)).toBe('████████')
  })

  test('a file caught mid-write, or missing its clock, is not a job', async () => {
    expect(parseJob('x.json', '{"label": "half')).toBe(null)
    expect(parseJob('x.json', '{"label": "no clock"}')).toBe(null)
  })

  test('durations read like a person would say them', async () => {
    expect(duration(42)).toBe('42s')
    expect(duration(907)).toBe('15m07s')
    expect(duration(6600)).toBe('1h50m')
  })
})

import { agentTail, median, summarize } from '../hooks/register'
import type { AgentRow } from '../types'

function agent(over: Partial<AgentRow> = {}): AgentRow {
  return { id: 'a1', type: 'deck-doctor', description: 'Diagnose edgar-vampires',
    status: 'running', startedAt: T0, endedAt: null, tools: 12, last: '',
    typical: null, runs: 0, late: false, ...over }
}

group('job-band agents', () => {
  test('an agent has no ETA, only its own history: "usually ~Xm, ~Ym to go"', async () => {
    const a = agent({ typical: 600, runs: 3 })
    expect(agentTail(a, T0 + 240)).toBe('4m00s elapsed · 12 tool calls · usually ~10m00s (3 runs), ~6m00s to go')
  })

  test('past its usual time it says so instead of a negative countdown', async () => {
    expect(agentTail(agent({ typical: 300, runs: 1 }), T0 + 420))
      .toBe('7m00s elapsed · 12 tool calls · past its usual ~5m00s by 2m00s')
  })

  test('an agent that started before the band loaded shows floors and no countdown', async () => {
    expect(agentTail(agent({ late: true, typical: 600, runs: 3 }), T0 + 120))
      .toBe('≥2m00s elapsed · ≥12 tool calls')
  })

  test('with no history it claims none', async () => {
    expect(agentTail(agent(), T0 + 61)).toBe('1m01s elapsed · 12 tool calls')
  })

  test('a finished agent reports how it ended and how long it took', async () => {
    expect(agentTail(agent({ status: 'completed', endedAt: T0 + 905, tools: 1 }), T0 + 950))
      .toBe('completed in 15m05s · 1 tool call')
  })

  test('the median is the middle run, not a mean a slow outlier drags', async () => {
    expect(median([300, 310, 3000])).toBe(310)
    expect(median([1, 3])).toBe(2)
    expect(median([])).toBe(null)
  })

  test('a tool call is said in a few words: its description first', async () => {
    expect(summarize({ tool: 'Bash', description: 'Run the try screen', command: 'x' }))
      .toBe('Bash: Run the try screen')
    expect(summarize({ tool: 'Read', file_path: '/a/b.json' })).toBe('Read: /a/b.json')
    expect(summarize({ tool: 'Bash', command: 'a\n   b' })).toBe('Bash: a b')
    expect(summarize({ tool: 'Bash', command: 'x'.repeat(200) }).length).toBe(70)
  })
})

import { isCruft, pidOf } from '../hooks/register'

group('job-band cruft', () => {
  test('the pid comes from the file name', async () => {
    expect(pidOf('simulate-54681.json')).toBe(54681)
    expect(pidOf('weird.json')).toBe(null)
  })

  test('a killed run is cruft; a slept machine is not', async () => {
    const killed = job({ updated_at: T0 + 10 })
    expect(isCruft({ ...killed, pid: 54681 }, T0 + 4000, false)).toBe(true)
    expect(isCruft({ ...killed, pid: 54681 }, T0 + 4000, true)).toBe(false)
  })

  test('a finished job is kept ten minutes, then removed', async () => {
    const done = job({ state: 'passed', updated_at: T0 })
    expect(isCruft(done, T0 + 599, true)).toBe(false)
    expect(isCruft(done, T0 + 601, true)).toBe(true)
  })
})
