import z from 'zod';

export const AuthStrategySchema = z.union([
  z.object({
    type: z.literal('apiKey'),
    value: z.string({
      description: 'API key for the API is required',
    }),
  }),
  z.object({
    type: z.literal('accessToken'),
    value: z.string({
      description: 'Access token for the API is required',
    }),
  }),
]);

export type AuthStrategy = z.infer<typeof AuthStrategySchema>;
