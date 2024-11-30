"use client";


import { AnimatePresence, motion } from "framer-motion";
import { ComponentRef, forwardRef } from "react";
import { useState, useEffect } from 'react';
import { BeatLoader } from 'react-spinners'
import { ArrowDownToLine, ThumbsUp, AudioLines, ChevronRight, Dot } from "lucide-react";
import { Tooltip } from 'react-tooltip'
import { cn } from "@/utils";
import { useVoice } from "./VoiceProvider";
import MicFFT from "./MicFFT";
import { generateEmptyFft } from './generateEmptyFft';

var texts = {
  agentName: process.env.NEXT_PUBLIC_AGENT_NAME || "感染力大师",
};


interface StreamingContentProps {
  content: string;
  speed?: number;
}

const StreamingContent: React.FC<StreamingContentProps> = ({ 
  content, 
  speed = 100
}) => {
  const [displayedContent, setDisplayedContent] = useState<string>('');
  const [currentIndex, setCurrentIndex] = useState<number>(0);

  useEffect(() => {
    if (currentIndex < content.length) {
      const timer = setTimeout(() => {
        setDisplayedContent(prev => prev + content[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, speed);

      return () => clearTimeout(timer);
    }
  }, [currentIndex, content, speed]);

  return (
    <span>
      {displayedContent}
    </span>
  );
};

const Messages = forwardRef<
  ComponentRef<typeof motion.div>,
  Record<never, never>
>(function Messages(_, ref) {
  const { messages, downloadAudio, replayAudio, fft, playedID, sendUserInput } = useVoice();
  const el = document.documentElement;
  const isDarkMode = el.classList.contains("dark");
  const emptyFft = generateEmptyFft();
  //console.log('messages', messages, Date.now())

  return (
    <motion.div
      layoutScroll
      className={"grow rounded-md overflow-auto p-4"}
      ref={ref}
    >
      <motion.div
        className={"max-w-2xl mx-auto w-full flex flex-col gap-2 pb-48"}
      >
        <AnimatePresence mode={"popLayout"}>
          {messages.map((msg, index) => {
            if (
              msg.type === "user_message" ||
              msg.type === "assistant_message"
            ) {
              // @ts-ignore
              const content = msg.message.content
              return (
                <motion.div
                  key={msg.type + index}
                  className={cn(
                    "w-[80%]",
                    "bg-card",
                    "border border-border rounded-md",
                    "mb-2",
                    msg.type === "user_message" ? "ml-auto" : ""
                  )}
                  initial={{
                    opacity: 0,
                    y: 10,
                  }}
                  animate={{
                    opacity: 1,
                    y: 0,
                  }}
                  exit={{
                    opacity: 0,
                    y: 0,
                  }}
                >
                  <div
                    className={cn(
                      "text-xs capitalize font-medium leading-none opacity-50 pt-4 px-3"
                    )}
                  >
                    {msg.type == "user_message" ? "游客" : texts.agentName}
                  </div>
                  <div className="pb-1 px-3 flex flex-col">
                    {msg.type == "user_message" ? content : (<StreamingContent content={content || ""} />)}
                    {msg.type == "assistant_message" && (
                    <div className={"flex flex-row h-12 w-4/5 p-1 items-center border rounded-lg"} style={{ marginTop: '0.5rem', marginBottom: '0.5rem'}}>
                      <AudioLines 
                        onClick={() => {replayAudio(msg.id || "")}}
                        size={20} className="rounded-md shadow-md hover:bg-muted cursor-pointer" style={{ marginRight: '0.5rem' , marginLeft: '0.5rem'}}
                      />
                      <div className={"relative grid h-8 w-40 shrink grow-0"}>
                        <MicFFT fft={playedID === msg.id ? fft : emptyFft} className={"fill-current"} />
                      </div>
                    </div>
                    )}
                    {msg.type == "assistant_message" && (
                    <div className="mt-auto ml-auto flex justify-between opacity-80">
                        <Tooltip id="my-tooltip" />
                        <ArrowDownToLine 
                          onClick={() => {downloadAudio(msg.id || "")}} 
                          className="hover:bg-muted cursor-pointer"
                          size={14} style={{ marginRight: '0.75rem' }} data-tooltip-id="my-tooltip" data-tooltip-content="下载" data-tooltip-place="down"
                        />
                        <ThumbsUp
                          className="hover:bg-muted cursor-pointer"
                          size={14} data-tooltip-id="my-tooltip" data-tooltip-content="赞" data-tooltip-place="down"
                        />
                    </div>
                    )}
                  </div>
                </motion.div>
              );
            } else if (msg.type === "post_assistant_message") {
              // @ts-ignore
              const content = msg.message.content
              // @ts-ignore
              const { q1, q2, q3 } = JSON.parse(content)
              return [q1, q2, q3].map((q, i) => (
                <motion.div
                  key={msg.type + index + i}
                  className={cn(
                    "w-[55%] sm:w-[35%]",
                    "bg-muted",
                    "border border-border rounded-xl",
                  )}
                  initial={{
                    opacity: 0,
                    y: 10,
                  }}
                  animate={{
                    opacity: 1,
                    y: 0,
                  }}
                  exit={{
                    opacity: 0,
                    y: 0,
                  }}
                >
                <div className="flex items-center cursor-pointer" 
                  onClick={() => {
                    sendUserInput(q)
                  }}
                >
                  <Dot size={20}/>
                  <div className="flex-1 mx-1 text-xs">{q}</div>
                  <ChevronRight size={16}/>
                </div>
                </motion.div>
              ));
            } else if (msg.type === "user_vad_message" && index === messages.length - 1) {
              return (
                <motion.div
                  key={'vad' + index}
                  className={cn(
                    "w-[80%]",
                    "bg-card",
                    "border border-border rounded-md",
                    "mb-2",
                    "ml-auto"
                  )}
                  initial={{
                    opacity: 0,
                    y: 10,
                  }}
                  animate={{
                    opacity: 1,
                    y: 0,
                  }}
                  exit={{
                    opacity: 0,
                    y: 0,
                  }}
                >
                  <div
                    className={cn(
                      "text-xs capitalize font-medium leading-none opacity-50 pt-4 px-3"
                    )}
                  >
                    {"游客"}
                  </div>
                  <div className={"pb-3 px-3"}>
                    <BeatLoader color={isDarkMode ? "#fff" : "#000"} size={8} />
                  </div>
                </motion.div>
              );
            } else if (msg.type === "assistant_notend_message" && index === messages.length - 1) {
              return (
                <motion.div
                  key={'notend' + index}
                  className={cn(
                    "w-[80%]",
                    "bg-card",
                    "border border-border rounded-md",
                    "mb-2",
                  )}
                  initial={{
                    opacity: 0,
                    y: 10,
                  }}
                  animate={{
                    opacity: 1,
                    y: 0,
                  }}
                  exit={{
                    opacity: 0,
                    y: 0,
                  }}
                >
                  <div
                    className={cn(
                      "text-xs capitalize font-medium leading-none opacity-50 pt-4 px-3"
                    )}
                  >
                    {texts.agentName}
                  </div>
                  <div className={"pb-3 px-3"}>
                    <BeatLoader color={isDarkMode ? "#fff" : "#000"} size={8} />
                  </div>
                </motion.div>
              );
            }

            return null;
          })}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
});

export default Messages;
