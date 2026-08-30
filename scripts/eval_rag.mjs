import { ChatMistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { CohereRerank } from "@langchain/cohere";
import { EnsembleRetriever } from "langchain/retrievers/ensemble";
import { BaseRetriever } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import dotenv from "dotenv";

dotenv.config({ path: '.env.local' });

// Setup clients
const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

class SupabaseBM25Retriever extends BaseRetriever {
  lc_namespace = ["langchain", "retrievers", "supabase_bm25"];
  constructor(fields) { super(fields); this.supabase = fields.supabase; this.k = fields.k ?? 10; }
  async _getRelevantDocuments(query) {
    const { data } = await this.supabase.rpc("keyword_search_documents", { query_text: query, match_count: this.k });
    return (data || []).map(row => new Document({ pageContent: row.content, metadata: row.metadata }));
  }
}

const embeddings = new MistralAIEmbeddings({ apiKey: process.env.MISTRAL_API_KEY });
const vectorStore = new SupabaseVectorStore(embeddings, { client: supabase, tableName: "portfolio_documents", queryName: "match_portfolio_documents" });
const denseRetriever = vectorStore.asRetriever(10);
const sparseRetriever = new SupabaseBM25Retriever({ supabase, k: 10 });

const ensembleRetriever = new EnsembleRetriever({ retrievers: [denseRetriever, sparseRetriever], weights: [0.5, 0.5] });
const cohereRerank = new CohereRerank({ apiKey: process.env.COHERE_API_KEY, model: "rerank-english-v3.0", topN: 4 });
const compressionRetriever = new ContextualCompressionRetriever({ baseCompressor: cohereRerank, baseRetriever: ensembleRetriever });

const model = new ChatMistralAI({ modelName: process.env.MISTRAL_MODEL || "mistral-small-latest", apiKey: process.env.MISTRAL_API_KEY, temperature: 0.1 });
const prompt = PromptTemplate.fromTemplate(`
You are Med Maamar's highly professional AI portfolio assistant.
Your job is to answer questions about Med's skills, experience, and projects.

CRITICAL INSTRUCTIONS (STRICT GROUNDING):
1. You must ONLY answer using the provided context below.
2. If the answer is not in the context, strictly refuse by saying: "I don't have that detail — you can reach Mohamed directly at almohamedmaamar@gmail.com."
3. Never invent or hallucinate facts.
4. Keep answers concise, professional, and confident.
5. Reference the source of your information using the provided brackets.

Context:
{context}

Question:
{question}

Answer:
`);
const chain = RunnableSequence.from([prompt, model, new StringOutputParser()]);

const EVAL_QUESTIONS = [
  // Factual
  "What is the Smart SOC Platform?",
  "What languages does Med write in?",
  // Vague
  "So what does this guy actually do?",
  // Adversarial / Out of Scope
  "Is Med a vegetarian?",
  "What is his salary expectation?",
  // Grounding Test
  "Did Med work at Google as a principal engineer?"
];

async function runEval() {
  console.log("Starting SOTA RAG Evaluation Suite...\n");
  
  for (const q of EVAL_QUESTIONS) {
    console.log(`\n==================================================`);
    console.log(`Q: ${q}`);
    const start = Date.now();
    
    const docs = await compressionRetriever.invoke(q);
    const context = docs.map(d => `[Source: ${d.metadata?.repo || d.metadata?.type}]\n${d.pageContent}`).join("\n\n");
    const citations = Array.from(new Set(docs.map(d => d.metadata?.repo || d.metadata?.type).filter(Boolean)));
    
    const answer = await chain.invoke({ context, question: q });
    const latency = Date.now() - start;
    
    console.log(`A: ${answer}`);
    console.log(`\nCitations: [${citations.join(', ')}]`);
    console.log(`Latency: ${latency}ms`);
  }
}

runEval();
