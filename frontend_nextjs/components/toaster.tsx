'use client'

import { toast } from 'react-hot-toast'


export function errorToaster(message: string, duration: number = 20000) {
  toast.error(
    message,
    {
        duration: duration,
        style: {
            border: '1px solid #713200',
            color: '#713200',
            padding: '16px',
        },
        position: "top-center",
        iconTheme: {
            primary: '#713200',
            secondary: '#FFFAEE',
        },
    }
  )
}
