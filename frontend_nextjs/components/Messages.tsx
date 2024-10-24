"use client";


import { AnimatePresence, motion } from "framer-motion";
import { ComponentRef, forwardRef, useState } from "react";
import { BeatLoader } from 'react-spinners'
import { ArrowDownToLine, ThumbsUp, AudioLines } from "lucide-react";
import { Tooltip } from 'react-tooltip'
import { cn } from "@/utils";
import { useVoice } from "./VoiceProvider";
import MicFFT from "./MicFFT";
import { generateEmptyFft } from './generateEmptyFft';

var texts = {
  agentName: process.env.NEXT_PUBLIC_AGENT_NAME || "感染力大师",
};

const Messages = forwardRef<
  ComponentRef<typeof motion.div>,
  Record<never, never>
>(function Messages(_, ref) {
  const { messages, downloadAudio, replayAudio, fft, playedID } = useVoice();
  const el = document.documentElement;
  const isDarkMode = el.classList.contains("dark");
  const emptyFft = generateEmptyFft();

  return (
    <motion.div
      layoutScroll
      className={"grow rounded-md overflow-auto p-4"}
      ref={ref}
    >
      <motion.div
        className={"max-w-2xl mx-auto w-full flex flex-col gap-4 pb-24"}
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
                    {content}
                    {msg.type == "assistant_message" && (
                    <div className={"flex flex-row h-12 w-4/5 p-1 items-center border rounded-lg"} style={{ marginTop: '0.5rem', marginBottom: '0.5rem'}}>
                      <AudioLines 
                        onClick={() => {replayAudio(msg.id || "")}}
                        size={20} className="rounded-md shadow-md hover:bg-muted" style={{ marginRight: '0.5rem' , marginLeft: '0.5rem'}}
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
                          className="hover:bg-muted"
                          size={14} style={{ marginRight: '0.75rem' }} data-tooltip-id="my-tooltip" data-tooltip-content="下载" data-tooltip-place="down"
                        />
                        <ThumbsUp
                          className="hover:bg-muted"
                          size={14} data-tooltip-id="my-tooltip" data-tooltip-content="赞" data-tooltip-place="down"
                        />
                    </div>
                    )}
                  </div>
                </motion.div>
              );
            } else if (msg.type === "user_vad_message" && index === messages.length - 1) {
              return (
                <motion.div
                  key={'vad' + index}
                  className={cn(
                    "w-[80%]",
                    "bg-card",
                    "border border-border rounded-md",
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
