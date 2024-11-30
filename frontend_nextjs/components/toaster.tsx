'use client'

import { toast } from 'react-hot-toast'


export function errorToaster(message: string, duration: number = 2000) {
  const el = document.documentElement;
  const dark = el.classList.contains("dark");
  const lightStyle = {
    border: '1px solid #713200',
    color: '#713200',
    padding: '16px',
  }
  const darkStyle = {
      borderRadius: '10px',
      background: '#333',
      color: '#fff',
  }
  toast.error(
    message,
    {
        duration: duration,
        style: dark ? darkStyle : lightStyle,
        position: "bottom-center",
        iconTheme: {
            primary: '#713200',
            secondary: '#FFFAEE',
        },
    }
  )
}
