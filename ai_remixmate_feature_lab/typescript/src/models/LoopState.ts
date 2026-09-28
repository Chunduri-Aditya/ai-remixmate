export interface LoopState {
  active: boolean;
  startSec: number;
  endSec: number;
  lengthBeats: number;
  quantized?: boolean;
  sourceCueId?: string;
}
