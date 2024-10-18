"use client";


import { useState, useEffect } from 'react';
import { useVoice } from "./VoiceProvider";
import { Button } from "./ui/button";
import { Mic, MicOff, Phone } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { Toggle } from "./ui/toggle";
import MicFFT from "./MicFFT";
import { ExternalLink } from './external-link'
import { cn } from "@/utils";

export default function Controls() {
  const [isClicked, setIsClicked] = useState(false);
  const { disconnect, status, isMuted, unmute, mute, micFft } = useVoice();

  useEffect(() => {
    setIsClicked(false);
  }, [status.value]);

  return (
    <div
      className={
        cn(
          "fixed bottom-0 left-0 w-full p-2 pb-1 flex items-center justify-center",
          "bg-gradient-to-t from-card via-card/90 to-card/0",
        )
      }
    >
      <AnimatePresence>
        {status.value === "connected" ? (
          <div className={"flex flex-col items-center"}>
          <motion.div
            initial={{
              y: "100%",
              opacity: 0,
            }}
            animate={{
              y: 0,
              opacity: 1,
            }}
            exit={{
              y: "100%",
              opacity: 0,
            }}
            className={
              "p-4 bg-card border border-border rounded-lg shadow-sm flex items-center gap-1"
            }
          >
            <Toggle
              pressed={!isMuted}
              onPressedChange={() => {
                if (isMuted) {
                  unmute();
                } else {
                  mute();
                }
              }}
            >
              {isMuted ? (
                <MicOff className={"size-4"} />
              ) : (
                <Mic className={"size-4"} />
              )}
            </Toggle>

            <div className={"relative grid h-8 w-40 shrink grow-0"}>
              <MicFFT fft={micFft} className={"fill-current"} />
            </div>

            <Button
              className={"flex items-center gap-1 p-2 px-3"}
              onClick={() => {
                setTimeout(() => {
                  disconnect();
                }, 2000)
                setIsClicked(true);
              }}
              style={{
                opacity: isClicked ? 0.5 : 1,
                transition: 'opacity 0.8s ease-in-out',
              }}
              variant={"destructive"}
            >
              <span>
                <Phone
                  className={"size-4 opacity-50"}
                  strokeWidth={2}
                  stroke={"currentColor"}
                />
              </span>
              <span>结束对话</span>
            </Button>
          </motion.div>
    <p
      className={cn(
        'p-2 text-center text-xs leading-normal text-muted-foreground',
      )}
    >
      对话由 AI 生成，请谨慎对待。
      <ExternalLink href="https://github.com/opendilab">
        OpenDILab
      </ExternalLink>
       开源项目 © 2024
    </p>
    </div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
