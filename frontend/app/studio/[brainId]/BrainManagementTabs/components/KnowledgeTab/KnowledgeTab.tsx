"use client";
import { UUID } from "crypto";
import { AnimatePresence, motion } from "framer-motion";

import { MessageInfoBox } from "@/lib/components/ui/MessageInfoBox/MessageInfoBox";
import Spinner from "@/lib/components/ui/Spinner";

import { KnowledgeTable } from "./KnowledgeTable/KnowledgeTable";
import { useAddedKnowledge } from "./hooks/useAddedKnowledge";

type KnowledgeTabProps = {
  brainId: UUID;
  hasEditRights: boolean;
};
export const KnowledgeTab = ({ brainId }: KnowledgeTabProps): JSX.Element => {
  const { isPending, allKnowledge } = useAddedKnowledge({
    brainId,
  });

  if (isPending) {
    return <Spinner />;
  }

  if (allKnowledge.length === 0) {
    return <MessageInfoBox type="info" content="hey" />;
  }

  return (
    <motion.div layout className="w-full flex flex-col gap-5">
      <AnimatePresence mode="popLayout">
        <KnowledgeTable knowledgeList={allKnowledge} />
      </AnimatePresence>
    </motion.div>
  );
};