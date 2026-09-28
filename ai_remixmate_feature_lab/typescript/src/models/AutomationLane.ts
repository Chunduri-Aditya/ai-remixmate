export interface AutomationPoint {
  timeSec: number;
  value: number;
}

export interface AutomationLane {
  id: string;
  target: string;
  interpolation: "linear" | "hold";
  points: AutomationPoint[];
}
