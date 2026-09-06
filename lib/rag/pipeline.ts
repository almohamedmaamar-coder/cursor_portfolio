import { getSupabaseAdmin } from "./db";
import { buildDenseRetriever } from "./retrievers";
import { reciprocalRankFusion } from "./retrievers";
import { generateQueryVariations } from "./multi_query";
import { rerank, generate, isSmallTalk, GREETING_RESPONSE } from "./generator";
import type { RagResult } from "./types";

interface ChatMessage { role: string; content: string; }

export async function runRagPipeline(query: string, history?: ChatMessage[]): Promise<RagResult> {
  if (isSmallTalk(query)) {
    return { content: GREETING_RESPONSE, citations: [] };
  }

  console.log("🔍 [RAG] Input query:", query);

  const supabase = getSupabaseAdmin();
  
  // Generate query variations using HyDE
  const queryVariations = await generateQueryVariations(query);
  console.log("🔄 [RAG] Query variations:", queryVariations);

  // Retrieve with each query variation
  const denseRetriever = buildDenseRetriever(supabase, 50);
  const retrievedLists = await Promise.all(
    queryVariations.map((q) => denseRetriever.invoke(q))
  );
  console.log("📄 [RAG] Retrieved lists per query:", retrievedLists.map((l) => l.length));

  // Merge with RRF
  const docs = reciprocalRankFusion(retrievedLists, 50);
  
  console.log("📄 [RAG] After RRF fusion count:", docs.length);
  console.log("📄 [RAG] Doc titles:", docs.map((d) => d.metadata?.title));
  console.log("📄 [RAG] First doc content preview:", docs[0]?.pageContent?.slice(0, 100));

  if (docs.length === 0) {
    return { content: "I don't have that detail — reach Med directly at almohamedmaamar@gmail.com.", citations: [] };
  }

  // Rerank all fused documents (top 10 chunks to prevent context bloat and rate limits)
  const rerankedDocs = await rerank(docs, query, Math.min(docs.length, 10));
  
  console.log("🎯 [RAG] Reranked docs count:", rerankedDocs.length);
  console.log("🎯 [RAG] Reranked titles:", rerankedDocs.map((d) => d.metadata?.title));
  console.log("🎯 [RAG] Reranked first 3 previews:", rerankedDocs.slice(0, 3).map((d) => d.pageContent?.slice(0, 80)));
  
  const result = await generate(rerankedDocs, query, history);

  return {
    ...result,
    rawContexts: rerankedDocs.map((d) => d.pageContent),
  };
}
