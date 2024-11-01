'use client'


import Card from "./Card";
import { useVoice } from "./VoiceProvider";
import { errorToaster } from "./toaster";
import { Toaster } from 'react-hot-toast';

const cards = [
  {
    title: "Speech-to-Speech Chat",
    description: "与感染力大师趣味畅聊~",
    icon: "/icons/hat1.jpg",
  },
  {
    title: "Advanced VC/TTS",
    description: "高保真的人声转换和文本转语音",
    icon: "/icons/hat1.jpg",
  },
  {
    title: "Streaming VC (WIP)",
    description: "实时流式声音转换和美化",
    icon: "/icons/hat1.jpg",
  }
];

export default function CardPanel() {
  const { connect } = useVoice();
  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 md:gap-6 max-w-6xl mx-auto">
        <Toaster />
        <div className="col-span-2 md:col-span-2 transition-all duration-300"
            onClick={() => {
                const sendHostname = process.env.NEXT_PUBLIC_FX_SEND_HOST || ""
                const recvHostname = process.env.NEXT_PUBLIC_FX_RECV_HOST || ""
                if (!sendHostname) {
                  errorToaster("请先指定正确的 NEXT_PUBLIC_FX_SEND_HOST 和 NEXT_PUBLIC_FX_RECV_HOST 环境变量")
                  return
                }
                connect({sendHostname, recvHostname, chatMode: true})
                .then(() => {})
                .catch(() => {})
                .finally(() => {});
            }}
        >
          <Card {...cards[0]} />
        </div>
        
        <div className="col-span-1 md:col-start-3 transition-all duration-300"
            onClick={() => {
                const sendHostname = process.env.NEXT_PUBLIC_VCTTS_SEND_HOST || ""
                const recvHostname = process.env.NEXT_PUBLIC_VCTTS_RECV_HOST || ""
                if (!sendHostname) {
                  errorToaster("请先指定正确的 NEXT_PUBLIC_VCTTS_SEND_HOST 和 NEXT_PUBLIC_VCTTS_RECV_HOST 环境变量")
                  return
                }
                connect({sendHostname, recvHostname, chatMode: false})
                .then(() => {})
                .catch(() => {})
                .finally(() => {});
            }}
        >
          <Card {...cards[1]} />
        </div>
        
        <div className="col-span-1 md:col-span-3 transition-all duration-300"
            onClick={() => {
                errorToaster("正在开发中，敬请期待")
            }}
        >
          <Card {...cards[2]} />
        </div>
      </div>
      <div className="flex mt-4 text-md text-primary font-medium justify-center">
        <span className="animate-bounce">点击任意模式开始</span>
      </div>
    </div>
  );
}
