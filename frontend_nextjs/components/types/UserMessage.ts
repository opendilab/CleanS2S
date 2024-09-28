import { ChatMessage } from './ChatMessage';

export interface MillisecondInterval {
  /** Start time of the interval in milliseconds. */
  begin: number;
  /** End time of the interval in milliseconds. */
  end: number;
}

export interface UserMessage {
  /**
   * The type of message sent through the socket; for a User Message, this must be `user_message`.
   *
   * This message contains both a transcript of the user’s input and the expression measurement predictions if the input was sent as an [Audio Input message](/reference/empathic-voice-interface-evi/chat/chat#send.Audio%20Input.type). Expression measurement predictions are not provided for a [User Input message](/reference/empathic-voice-interface-evi/chat/chat#send.User%20Input.type), as the prosody model relies on audio input and cannot process text alone.
   */
  type: 'user_message' | 'user_vad_message';
  /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
  customSessionId?: string;
  /** Transcript of the message. */
  message: ChatMessage;
  /** Start and End time of user message. */
  time?: MillisecondInterval;
  /** Indicates if this message was inserted into the conversation as text from a [User Input](/reference/empathic-voice-interface-evi/chat/chat#send.User%20Input.text) message. */
  fromText: boolean;
  receivedAt?: Date;
}
