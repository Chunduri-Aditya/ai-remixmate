export interface CrossfadeGains {
  leftGain: number;
  rightGain: number;
}

export function clampCrossfader(position: number): number {
  if (!Number.isFinite(position)) return 0;
  return Math.max(-1, Math.min(1, position));
}

export function linearCrossfade(position: number): CrossfadeGains {
  const p = clampCrossfader(position);
  return {
    leftGain: (1 - p) / 2,
    rightGain: (1 + p) / 2
  };
}

export function equalPowerCrossfade(position: number): CrossfadeGains {
  const p = (clampCrossfader(position) + 1) / 2;
  return {
    leftGain: Math.cos(p * Math.PI / 2),
    rightGain: Math.sin(p * Math.PI / 2)
  };
}
