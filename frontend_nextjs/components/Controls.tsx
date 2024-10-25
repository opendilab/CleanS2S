"use client";


import { useState, useEffect, useRef, ChangeEvent } from 'react';
import Textarea from 'react-textarea-autosize'
import { useVoice } from "./VoiceProvider";
import { Button } from "./ui/button";
import { Mic, MicOff, Volume2, VolumeX, Phone, BadgePlus, MessageSquareMore, CornerDownLeft } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { Toggle } from "./ui/toggle";
import MicFFT from "./MicFFT";
import { ExternalLink } from './external-link'
import { cn } from "@/utils";

export default function Controls() {
  const [isClicked, setIsClicked] = useState(false);
  const [enableTyping, setEnableTyping] = useState(false);
  const [color, setColor] = useState("currentColor");
  const [audioColor, setAudioColor] = useState("currentColor")
  // use useRef to avoid the "Maximum update depth exceeded" problem
  const prevAudioColorRef = useRef(audioColor);
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }, [])
  const handleInputChange = async (event: ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.target.value;
    setInput(value)
  }

  const { disconnect, status, isMuted, unmute, mute, micFft, isAudioMuted, unmuteAudio, muteAudio, fft, sendUserInput, clearCurrentTopic } = useVoice();

  useEffect(() => {
    setIsClicked(false);
  }, [status.value]);

  useEffect(() => {
    // micFFT is an array of 24 values
    if (micFft.some((v) => v > 1)) {
      setColor("#C1121F");
    } else {
      setColor("currentColor");
    }
  }, [micFft]);

  useEffect(() => {
    // fft is an array of 24 values
    if (fft.some((v) => v > 0.5)) {
      if (prevAudioColorRef.current != "#C1121F") {
        setAudioColor("#C1121F");
        prevAudioColorRef.current = "#C1121F"
      }
    } else {
      if (prevAudioColorRef.current != "currentColor") {
        setAudioColor("currentColor");
        prevAudioColorRef.current = "currentColor"
      }
    }
  }, [fft]);

  const handleNewTopic = () => {
    sendUserInput("new topic");
    clearCurrentTopic();
  } 

  const handleEnableTyping = () => {
    if (enableTyping) {
      unmute();
    } else {
      mute();
    }
    setEnableTyping(!enableTyping);
  }

  const handleSubmit = () => {
    if (input.length > 0) {
      sendUserInput(input);
      setInput("");
    }
  };

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
          <div className="flex flex-row items-center gap-4">
          <BadgePlus 
            onClick={handleNewTopic}
            className={"flex flex-row size-4 mb-2 justify-start border-none shadow-xl rounded-lg bg-accent h-8 w-8 p-1 cursor-pointer"}
          />
          <MessageSquareMore
            onClick={handleEnableTyping}
            className={"flex flex-row size-4 mb-2 justify-start border-none shadow-xl rounded-lg bg-accent h-8 w-8 p-1 cursor-pointer"}
          />
          </div>
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
          <div className="flex flex-col items-center gap-2">
            <div className="flex flex-row items-center gap-2">
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
                <Mic className={"size-4"} color={color}/>
              )}
            </Toggle>

            <Toggle
              pressed={!isAudioMuted}
              onPressedChange={() => {
                if (isAudioMuted) {
                  unmuteAudio();
                } else {
                  muteAudio();
                }
              }}
            >
              {isAudioMuted ? (
                <VolumeX className={"size-4"} />
              ) : (
                <Volume2 className={"size-4"} color={audioColor}/>
              )}
            </Toggle>

            <div className={"relative grid h-8 w-32 shrink grow-0"}>
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
            </div>

          {enableTyping && (
            <div className="relative flex max-h-60 w-full grow flex-col overflow-hidden bg-background pr-8 sm:rounded-md border">
              <Textarea
                ref={inputRef}
                tabIndex={0}
                rows={1}
                value={input}
                onChange={handleInputChange}
                placeholder={"一字胜万言~输入完成后点击按钮发送"}
                spellCheck={false}
                className="min-h-[60px] max-h-[90px] overflow-auto w-full resize-none bg-transparent px-4 py-[1.3rem] focus-within:outline-none sm:text-sm"
              />
              <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
                <CornerDownLeft
                  onClick={handleSubmit}
                  size={32}
                  color={"currentColor"}
                  className="bg-secondary p-2 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          )}
          </div>
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
