"use client";

import { AudioInput, AudioOutput, AssistantInput, SessionSettings, PauseAssistantMessage, ResumeAssistantMessage } from './types';
import { CloseEvent, ErrorEvent } from './events';
import { ReconnectingWebSocket } from './WebSocket';


export type PublishEvent = SessionSettings
export type SubscribeEvent = AudioOutput


export class CustomError extends Error {
  readonly statusCode?: number;
  readonly body?: unknown;

  constructor({
    message,
    statusCode,
    body,
  }: {
    message?: string;
    statusCode?: number;
    body?: unknown;
  }) {
    super(buildMessage({ message, statusCode, body }));
    Object.setPrototypeOf(this, CustomError.prototype);
    if (statusCode != null) {
      this.statusCode = statusCode;
    }

    if (body !== undefined) {
      this.body = body;
    }
  }
}

function buildMessage({
  message,
  statusCode,
  body,
}: {
  message: string | undefined;
  statusCode: number | undefined;
  body: unknown | undefined;
}): string {
  const lines: string[] = [];
  if (message != null) {
    lines.push(message);
  }

  if (statusCode != null) {
    lines.push(`Status code: ${statusCode.toString()}`);
  }

  if (body != null) {
    lines.push(`Body: ${JSON.stringify(body, undefined, 2)}`);
  }

  return lines.join('\n');
}



export declare namespace ChatSocket {
  interface Args {
    sendSocket: ReconnectingWebSocket;
    recvSocket: ReconnectingWebSocket;
  }

  type Response = SubscribeEvent & { receivedAt: Date };

  type EventHandlers = {
    open?: () => void;
    message?: (message: Response) => void;
    close?: (event: CloseEvent) => void;
    error?: (error: Error) => void;
  };
}

export class ChatSocket {
  public readonly sendSocket: ReconnectingWebSocket;
  public readonly recvSocket: ReconnectingWebSocket;
  public readonly sendReadyState: number;
  public readonly recvReadyState: number;

  protected readonly sendEventHandlers: ChatSocket.EventHandlers = {};
  protected readonly recvEventHandlers: ChatSocket.EventHandlers = {};

  private idCount: number;

  constructor({ sendSocket, recvSocket }: ChatSocket.Args) {
    this.sendSocket = sendSocket;
    this.recvSocket = recvSocket;
    this.sendReadyState = sendSocket.readyState;
    this.recvReadyState = recvSocket.readyState;

    this.sendSocket.addEventListener('open', this.handleSendOpen);
    this.sendSocket.addEventListener('message', this.handleSendMessage);
    this.sendSocket.addEventListener('close', this.handleSendClose);
    this.sendSocket.addEventListener('error', this.handleSendError);
    this.recvSocket.addEventListener('open', this.handleRecvOpen);
    this.recvSocket.addEventListener('message', this.handleRecvMessage);
    this.recvSocket.addEventListener('close', this.handleRecvClose);
    this.recvSocket.addEventListener('error', this.handleRecvError);
    this.idCount = 1
  }

  on<T extends keyof ChatSocket.EventHandlers>(
    event: T,
    callback: ChatSocket.EventHandlers[T],
  ) {
    if (event === 'message') {
      this.sendEventHandlers[event] = callback;
      this.recvEventHandlers[event] = callback;
    } else if (event === 'error') {
      this.sendEventHandlers[event] = callback;
      this.recvEventHandlers[event] = callback;
    } else {
      this.sendEventHandlers[event] = callback;
    }
  }

  /**
   * Send audio input
   */
  public sendAudioInput(
    message: Omit<AudioInput, 'type'>,
  ): void {
    this.assertSocketIsOpen();
    this.sendJson({
      type: 'audio_input',
      ...message,
    });
  }

  /**
   * Send session settings
   */
  public sendSessionSettings(
    message: Omit<SessionSettings, 'type'>,
  ): void {
    this.assertSocketIsOpen();
    this.sendJson({
      type: 'session_settings',
      ...message,
    });
  }

  /**
   * Send assistant input
   */
  public sendAssistantInput(
    message: Omit<AssistantInput, 'type'>,
  ): void {
    this.assertSocketIsOpen();
    this.sendJson({
      type: 'assistant_input',
      ...message,
    });
  }

  /**
   * Send pause assistant message
   */
  public pauseAssistant(
    message: Omit<PauseAssistantMessage, 'type'>,
  ): void {
    this.assertSocketIsOpen();
    this.sendJson({
      type: 'pause_assistant_message',
      ...message,
    });
  }

  /**
   * Send resume assistant message
   */
  public resumeAssistant(
    message: Omit<ResumeAssistantMessage, 'type'>,
  ): void {
    this.assertSocketIsOpen();
    this.sendJson({
      type: 'resume_assistant_message',
      ...message,
    });
  }

  /**
   * Send text input
   */
  public sendUserInput(text: string): void {
    this.assertSocketIsOpen();
    this.sendJson({
      type: 'user_input',
      text,
    });
  }

  /**
   * @name connect
   * @description
   * Connect to the ReconnectingWebSocket.
   */
  public connect(): ChatSocket {
    this.sendSocket.reconnect();
    this.recvSocket.reconnect();

    this.sendSocket.addEventListener('open', this.handleSendOpen);
    this.sendSocket.addEventListener('message', this.handleSendMessage);
    this.sendSocket.addEventListener('close', this.handleSendClose);
    this.sendSocket.addEventListener('error', this.handleSendError);
    this.recvSocket.addEventListener('open', this.handleRecvOpen);
    this.recvSocket.addEventListener('message', this.handleRecvMessage);
    this.recvSocket.addEventListener('close', this.handleRecvClose);
    this.recvSocket.addEventListener('error', this.handleRecvError);
    return this;
  }

  /**
   * Closes the underlying socket.
   */
  public close(): void {
    this.sendSocket.close();
    this.recvSocket.close();

    this.handleSendClose({ code: 1000 } as CloseEvent);

    this.sendSocket.removeEventListener('open', this.handleSendOpen);
    this.sendSocket.removeEventListener('message', this.handleSendMessage);
    this.sendSocket.removeEventListener('close', this.handleSendClose);
    this.sendSocket.removeEventListener('error', this.handleSendError);
    this.recvSocket.removeEventListener('open', this.handleRecvOpen);
    this.recvSocket.removeEventListener('message', this.handleRecvMessage);
    this.recvSocket.removeEventListener('close', this.handleRecvClose);
    this.recvSocket.removeEventListener('error', this.handleRecvError);
  }

  public async tillSocketOpen(): Promise<ReconnectingWebSocket> {
    if (this.sendSocket.readyState === ReconnectingWebSocket.OPEN) {
      return this.sendSocket;
    }
    return new Promise((resolve, reject) => {
      this.sendSocket.addEventListener('open', () => {
        resolve(this.sendSocket);
      });

      this.sendSocket.addEventListener('error', (event: unknown) => {
        reject(event);
      });
    });
  }

  private assertSocketIsOpen(): void {
    if (!this.sendSocket) {
      throw new CustomError({ message: 'Socket is not connected.' });
    }

    if (this.sendSocket.readyState !== ReconnectingWebSocket.OPEN) {
      throw new CustomError({ message: 'Socket is not open.' });
    }
  }

  private sendJson(payload: any): void {
    this.sendSocket.send(JSON.stringify(payload));
  }

  private handleSendOpen = () => {
    this.sendEventHandlers.open?.();
  };

  private handleRecvOpen = () => {
    this.recvEventHandlers.open?.();
  };

  private handleSendMessage = (event: { data: string }): void => {
    const jsonData = JSON.parse(event.data);
    if (jsonData?.return_info) {
      const timestamp = new Date().toISOString()
      // @ts-ignore
      const output: AudioOutput = {
        type: 'vad_output',
        id: String(this.idCount),
        data: new Int16Array(0),
        question: "",
        answer: "",
      };
      // @ts-ignore
      this.sendEventHandlers.message?.(output);
    }
  }

  private handleRecvMessage = (event: { data: string }): void => {
    const jsonData = JSON.parse(event.data);
    const audioBase64 = jsonData.answer_audio;
    const audioBytes = Uint8Array.from(atob(audioBase64), c => c.charCodeAt(0));
    const dataView = new DataView(audioBytes.buffer);
    const audioInt16 = new Int16Array(audioBytes.length / 2);
    
    for (let i = 0; i < audioInt16.length; i++) {
        audioInt16[i] = dataView.getInt16(i * 2, true);
    }

    const audioOutputObject: AudioOutput = {
        type: 'audio_output',
        id: String(this.idCount),
        data: audioInt16,
        question: jsonData.question_text,
        answer: jsonData.answer_text,
        end: jsonData.end_flag,
    };
    // @ts-ignore
    this.recvEventHandlers.message?.(audioOutputObject);
    this.idCount++;
  };

  private handleSendClose = (event: CloseEvent) => {
    this.sendEventHandlers.close?.(event);
  };

  private handleRecvClose = (event: CloseEvent) => {
    this.recvEventHandlers.close?.(event);
  };

  private handleSendError = (event: ErrorEvent) => {
    const message = event.message ?? 'ReconnectingWebSocket error';
    this.sendEventHandlers.error?.(new Error(message));
  };

  private handleRecvError = (event: ErrorEvent) => {
    const message = event.message ?? 'ReconnectingWebSocket error';
    this.recvEventHandlers.error?.(new Error(message));
  };
}
