export class SocketUnknownMessageError extends Error {
  constructor(message?: string) {
    super(`Unknown message type.${message ? ' ' + message : ''}`);
    this.name = 'SocketUnknownMessageError';
  }
}

/**
 * @name isSocketUnknownMessageError
 * @description
 * Check if an error is a SocketUnknownMessageError.
 * @param err - The error to check.
 * @returns
 * `true` if the error is a SocketUnknownMessageError.
 * @example
 * ```ts
 * if (isSocketUnknownMessageError(err)) {
 * console.error('Unknown message type');
 * }
 * ```
 */
export const isSocketUnknownMessageError = (
  err: unknown,
): err is SocketUnknownMessageError => {
  return err instanceof SocketUnknownMessageError;
};

export class SocketFailedToParseMessageError extends Error {
  constructor(message?: string) {
    super(
      `Failed to parse message from socket.${message ? ' ' + message : ''}`,
    );
    this.name = 'SocketFailedToParseMessageError';
  }
}

/**
 * @name isSocketFailedToParseMessageError
 * @description
 * Check if an error is a SocketFailedToParseMessageError.
 * @param err - The error to check.
 * @returns
 * `true` if the error is a SocketFailedToParseMessageError.
 * @example
 * ```ts
 * if (isSocketFailedToParseMessageError(err)) {
 * console.error('Failed to parse message from socket');
 * }
 * ```
 */
export const isSocketFailedToParseMessageError = (
  err: unknown,
): err is SocketFailedToParseMessageError => {
  return err instanceof SocketFailedToParseMessageError;
};
