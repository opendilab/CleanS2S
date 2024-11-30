"use client";


import { AnimatePresence, motion } from "framer-motion";
import { Phone } from "lucide-react";
import { useVoice } from "./VoiceProvider";
import { Button } from "./ui/button";
import CardPanel from "./CardPanel";

export default function StartCall() {
  const { status } = useVoice();

  return (
    <AnimatePresence>
      {status.value !== "connected" ? (
        <motion.div
          className={"fixed inset-0 p-4 flex items-center justify-center bg-background"}
          initial="initial"
          animate="enter"
          exit="exit"
          variants={{
            initial: { opacity: 0 },
            enter: { opacity: 1 },
            exit: { opacity: 0 },
          }}
        >
          <CardPanel />
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
