import dynamic from "next/dynamic";

const Chat = dynamic(() => import("@/components/Chat"), {
  ssr: false,
});

export default async function Page() {

  return (
    <div className={"grow flex flex-col"}>
      <Chat/>
    </div>
  );
}
