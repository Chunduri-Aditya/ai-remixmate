export interface MixerEqState {
  lowDb: number;
  midDb: number;
  highDb: number;
}

export interface MixerChannelState {
  id: string;
  deckId: string;
  volume: number;
  gainDb: number;
  eq: MixerEqState;
  mute?: boolean;
  solo?: boolean;
  meter?: number;
}

export interface MixerState {
  crossfader: number;
  masterGainDb: number;
  cueMix?: number;
  limiterEnabled?: boolean;
  channels: MixerChannelState[];
}
