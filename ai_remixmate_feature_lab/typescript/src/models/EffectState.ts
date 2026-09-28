export type EffectType = "filter" | "echo" | "reverb" | "gate" | "delay" | "none";

export interface EffectState {
  id: string;
  type: EffectType;
  enabled: boolean;
  wet: number;
  parameters: Record<string, number>;
}
