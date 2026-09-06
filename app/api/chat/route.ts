/**
 * app/api/chat/route.ts
 * Thin Next.js API route — delegates all logic to lib/rag/pipeline.
 */
import { NextRequest, NextResponse } from "next/server";
import { runRagPipeline } from "@/lib/rag/pipeline";
import { getSupabaseAdmin } from "@/lib/rag/db";
import { getOrCreateSession, logMessage } from "@/lib/rag/telemetry";

export async function POST(req: NextRequest) {
  try {
    const { messages, visitorFingerprint } = await req.json();
    const query: string = messages[messages.length - 1]?.content ?? "";
    const history = messages.slice(0, -1);
    const isEval = req.headers.get("x-eval-mode") === "true";

    if (!query.trim()) {
      return NextResponse.json({ error: "Empty query" }, { status: 400 });
    }

    const supabase = getSupabaseAdmin();

    const [result, sessionId] = await Promise.all([
      runRagPipeline(query, history),
      getOrCreateSession(supabase, visitorFingerprint || "anonymous-visitor"),
    ]);

    // Reliably log both messages to Postgres before returning
    if (sessionId) {
      await Promise.allSettled([
        logMessage(supabase, sessionId, "user", query),
        logMessage(supabase, sessionId, "assistant", result.content),
      ]);
    }

    return NextResponse.json({
      role: "assistant",
      content: result.content,
      citations: result.citations,
      ...(isEval && { rawContexts: result.rawContexts })
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[Chat API]", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
