export type Encoding = 'linear16';


export interface AudioConfiguration {
  /** Encoding format of the audio input, such as `linear16`. */
  encoding: Encoding;
  /** Number of audio channels. */
  channels: number;
  /** Audio sample rate. Number of samples per second in the audio input, measured in Hertz. */
  sampleRate: number;
}
