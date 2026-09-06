/**
 * lib/rag/telemetry.ts
 * Fire-and-forget telemetry logging for recruiter session tracking.
 * Never blocks the main response path.
 */
import type { SupabaseClient } from "@supabase/supabase-js";

export async function getOrCreateSession(
  supabase: SupabaseClient,
  visitorFingerprint: string
): Promise<string | null> {
  let { data: session } = await supabase
    .from("chat_sessions")
    .select("id")
    .eq("visitor_fingerprint", visitorFingerprint)
    .single();

  if (!session) {
    const { data: newSession } = await supabase
      .from("chat_sessions")
      .insert({ visitor_fingerprint: visitorFingerprint })
      .select("id")
      .single();
    session = newSession;
  }

  return session?.id ?? null;
}

export async function logMessage(
  supabase: SupabaseClient,
  sessionId: string,
  role: "user" | "assistant",
  content: string
): Promise<void> {
  try {
    const { error } = await supabase
      .from("chat_messages")
      .insert({ session_id: sessionId, role, content });
    if (error) {
      console.error(`❌ [Telemetry] Failed to log ${role} message:`, error.message);
    } else {
      console.log(`✅ [Telemetry] Saved ${role} message to Postgres [session: ${sessionId.slice(0, 8)}...]`);
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error(`❌ [Telemetry] Exception logging ${role} message:`, msg);
  }
}
