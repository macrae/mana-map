/** One job as its progress file states it (see register.tsx for the protocol). */
export type Job = {
  file: string
  label: string
  done: number
  total: number | null
  unit: string
  failed: number
  state: 'running' | 'passed' | 'failed'
  startedAt: number
  updatedAt: number
  detail: string
  /** From the file name `<name>-<pid>.json`; how a dead run is told from a stalled one. */
  pid: number | null
}

/** One subagent, as `$.agent.list()` and its own tool calls show it. */
export type AgentRow = {
  id: string
  type: string
  description: string
  status: string
  startedAt: number
  endedAt: number | null
  tools: number
  last: string
  /** Median seconds this agent type took on past runs, and how many runs. */
  typical: number | null
  runs: number
  /** Already running when the band loaded: start and tool count are floors. */
  late: boolean
  /** The response-time target in seconds, from `sla_s:` in the charter's frontmatter. */
  sla: number | null
}

export type Snapshot = { now: number; jobs: Job[]; agents: AgentRow[] }

declare module 'claude-code' {
  interface PluginState {
    'job-band': { snapshot: Snapshot; isHidden: boolean }
  }
}
