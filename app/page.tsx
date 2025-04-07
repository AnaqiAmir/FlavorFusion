"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ModeToggle } from "@/components/mode-toggle";
import { PlusCircle, Send, User, Bot } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Link from "next/link";
import ReactMarkdown from "react-markdown";

export default function Home() {
  // State variables
  // messages: array of message objects with role and content
  // input: string for user input
  // conversations: array of conversation objects
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const [conversations, setConversations] = useState<any[]>([]);

  // Fetch conversations from the server

  // async function fetchConversations() {
  //   try {
  //     const res = await fetch("/api/conversations");
  //     if (!res.ok) {
  //       throw new Error("Failed to fetch conversations");
  //     }
  //     const data = await res.json();
  //     setConversations(data);
  //   } catch (error) {
  //     console.error(error);
  //   }
  // }

  // // Fetch conversations when the component mounts
  // useEffect(() => {
  //   fetchConversations();
  // }, []);

  // Function to handle new chat creation
  // async function handleNewChat() {
  //   try {
  //     const res = await fetch("/api/conversations", {
  //       method: "POST",
  //     });
  //     if (!res.ok) {
  //       throw new Error("Failed to create new chat");
  //     }
  //     fetchConversations();
  //   } catch (error) {
  //     console.error(error);
  //   }
  // }

  // Function to handle sending messages
  const handleSend = async () => {
    if (!input.trim()) return;

    // Push user message
    setMessages((prev) => [...prev, { role: "user", content: input }]);

    try {
      const res = await fetch("http://127.0.0.1:5328/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: input }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();

      // Push assistant response
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (err) {
      console.error("Error sending message:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setInput("");
    }
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar (Left Nav) */}
      <div className="hidden md:flex md:flex-col w-60 bg-muted p-4">
        <div className="flex-1 overflow-y-auto space-y-4">
          <h2 className="text-lg font-bold">Chats</h2>
          {/* Example chat history items */}
          <Button variant="ghost" className="justify-start w-full">
            Conversation 1
          </Button>
          <Button variant="ghost" className="justify-start w-full">
            Conversation 2
          </Button>
          <Button variant="ghost" className="justify-start w-full">
            Conversation 3
          </Button>
          {/* {conversations.map((conv) => (
            <Button
              key={conv.id}
              variant="ghost"
              className="justify-start w-full"
              onClick={() => {
                // logic to open or switch to this conversation
              }}
            >
              {conv.title}
            </Button>
          ))} */}
        </div>
        <div className="mt-4 space-y-2 border-t pt-4">
          <Button
            variant="outline"
            className="w-full justify-center"
            onClick={() => {}}
          >
            <PlusCircle className="mr-2 h-4 w-4" />
            New Chat
          </Button>
          <ModeToggle />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col bg-background">
        <header className="border-b p-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Chat Topic</h1>
          <div className="flex items-center space-x-4">
            <Link href="/profile">
              <Avatar>
                <AvatarFallback>
                  <User className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
            </Link>
          </div>
        </header>
        {/* Chat Section */}
        <ScrollArea className="flex-1 p-4">
          <div className="flex flex-col space-y-4">
            {messages.map((message, index) => {
              const isUser = message.role === "user";
              return (
                <div
                  key={index}
                  className={isUser ? "flex justify-end" : "flex justify-start"}
                >
                  {/* Bubble */}
                  <div
                    className={`prose max-w-md rounded-lg px-4 py-2 text-sm mb-2 ${
                      isUser
                        ? "bg-accent text-accent-foreground"
                        : "bg-background text-foreground border border-muted"
                    }`}
                  >
                    {isUser ? (
                      // Show user messages as plain text
                      <>{message.content}</>
                    ) : (
                      // Show assistant messages as markdown
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>

        {/* Input Section */}
        <div className="p-4 border-t">
          <div className="flex items-center space-x-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSend();
                  setInput("");
                }
              }}
              placeholder="Send a message..."
            />
            <Button
              variant="default"
              onClick={() => {
                handleSend();
                setInput("");
              }}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
