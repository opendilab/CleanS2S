export interface AudioInput {
  /**
   * The type of message sent through the socket; must be `audio_input` for our server to correctly identify and process it as an Audio Input message.
   *
   * This message is used for sending audio input data to EVI for processing and expression measurement. Audio data should be sent as a continuous stream, encoded in Base64.
   */
  type: 'audio_input';
  /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
  customSessionId?: string;
  /**
   * Base64 encoded audio input to insert into the conversation.
   *
   * The content of an Audio Input message is treated as the user’s speech to EVI and must be streamed continuously. Pre-recorded audio files are not supported.
   *
   * For optimal transcription quality, the audio data should be transmitted in small chunks.
   */
  data: string;
}
