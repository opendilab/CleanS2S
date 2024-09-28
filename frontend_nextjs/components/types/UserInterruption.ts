export interface UserInterruption {
    /**
     * The type of message sent through the socket; for a User Interruption message, this must be `user_interruption`.
     *
     * This message indicates the user has interrupted the assistant’s response. EVI detects the interruption in real-time and sends this message to signal the interruption event. This message allows the system to stop the current audio playback, clear the audio queue, and prepare to handle new user input.
     */
    type: "user_interruption";
    /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
    customSessionId?: string;
    /** Unix timestamp of the detected user interruption. */
    time: number;
}
