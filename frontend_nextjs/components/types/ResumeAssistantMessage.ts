export interface ResumeAssistantMessage {
    /**
     * The type of message sent through the socket; must be `resume_assistant_message` for our server to correctly identify and process it as a Resume Assistant message.
     *
     * Upon resuming, if any audio input was sent during the pause, EVI will retain context from all messages sent but only respond to the last user message. (e.g., If you ask EVI two questions while paused and then send a `resume_assistant_message`, EVI will respond to the second question and have added the first question to its conversation context.)
     */
    type: "resume_assistant_message";
    /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
    customSessionId?: string;
}
