export interface PauseAssistantMessage {
  /**
   * The type of message sent through the socket; must be `pause_assistant_message` for our server to correctly identify and process it as a Pause Assistant message.
   *
   * Once this message is sent, EVI will not respond until a [Resume Assistant message](/reference/empathic-voice-interface-evi/chat/chat#send.Resume%20Assistant%20Message.type) is sent. When paused, EVI won’t respond, but transcriptions of your audio inputs will still be recorded.
   */
  type: 'pause_assistant_message';
  /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
  customSessionId?: string;
}
