import z from 'zod';

export const AudioMessageSchema = z
  .object({
    type: z.literal('audio'),
    data: z.instanceof(ArrayBuffer),
  })
  .transform((obj) => {
    return Object.assign(obj, {
      receivedAt: new Date(),
    });
  });

export type AudioMessage = z.infer<typeof AudioMessageSchema>;

export const parseAudioMessage = async (
  blob: Blob,
): Promise<AudioMessage | null> => {
  return blob
    .arrayBuffer()
    .then((buffer) => {
      return {
        type: 'audio' as const,
        data: buffer,
        receivedAt: new Date(),
      };
    })
    .catch(() => {
      return null;
    });
};
