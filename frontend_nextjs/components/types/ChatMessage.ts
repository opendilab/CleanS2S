export interface ChatMessage {
  /** Role of who is providing the message. */
  role: string;
  /** Transcript of the message. */
  content?: string;
}
