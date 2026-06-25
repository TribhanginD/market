"""
Base Claude agent class.
Implements the Anthropic agentic loop:
  1. Send message with tools → 2. If tool_use → execute → return result → repeat
  3. If end_turn → return final text answer
"""

import json
import logging
import time
from typing import Any, Optional

import anthropic

import config
from llm.providers import GroqProvider, GeminiProvider, MistralProvider, parse_model_provider
from llm.providers import OpenAICompatProvider

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all pipeline agents.
    Each agent wraps a Claude model with:
    - A system prompt defining its persona/role
    - A set of tools it can call
    - An agentic loop that runs until the model produces a final answer
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict] = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        max_iterations: int = 15,
    ):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.system = system_prompt
        
        # Inject constraint if Caveman mode enabled
        if getattr(config, "CAVEMAN_MODE", False):
            caveman_rules = (
                "\n\nCRITICAL TOKEN OPTIMIZATION RULE: CAVEMAN MODE ACTIVE.\n"
                "You must speak like a caveman. Drop all prepositions, conversational filler, niceties, and complete sentences.\n"
                "Output ONLY raw technical logic. Use the absolute minimum words. 0% fluff, 100% substance.\n"
                "Example: Instead of 'Based on the P/E of 50, it appears overvalued', output 'P/E 50. Overvalued.'\n"
            )
            self.system += caveman_rules
            
        self.tools = tools or []
        self.model = model or config.MODEL_SMART
        self.provider_name, self.provider_model = parse_model_provider(self.model)
        # Providers instantiated lazily — avoids validating unused API keys on import.
        self._groq_inst = None
        self._gemini_inst = None
        self._mistral_inst = None
        self._openai_inst = None
        self.max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.max_iterations = max_iterations
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def _groq(self):
        if self._groq_inst is None:
            self._groq_inst = GroqProvider()
        return self._groq_inst

    @property
    def _gemini(self):
        if self._gemini_inst is None:
            self._gemini_inst = GeminiProvider()
        return self._gemini_inst

    @property
    def _mistral(self):
        if self._mistral_inst is None:
            self._mistral_inst = MistralProvider()
        return self._mistral_inst

    @property
    def _openai(self):
        if self._openai_inst is None:
            self._openai_inst = OpenAICompatProvider()
        return self._openai_inst

    def run(self, user_message: str, context: dict = None) -> str:
        """
        Run the agent on a user message.
        
        Args:
            user_message: The task for this agent
            context: Optional dict — will be serialized and prepended to message
            
        Returns:
            Final text response from the model
        """
        # Build initial message
        if context:
            context_str = json.dumps(context, indent=2, default=str)
            full_message = f"<context>\n{context_str}\n</context>\n\n{user_message}"
        else:
            full_message = user_message
        
        messages = [{"role": "user", "content": full_message}]
        
        # Non-Anthropic providers
        if self.provider_name == "gemini":
            if self.tools:
                return self._run_gemini_tool_loop(messages)
            return self._run_simple_provider(messages)
        if self.provider_name in ("groq", "mistral", "openai"):
            if self.tools:
                raise RuntimeError(
                    f"Provider '{self.provider_name}' does not support tool loop. Use gemini or anthropic."
                )
            return self._run_simple_provider(messages)

        # Agentic loop (Anthropic + tools)
        for iteration in range(self.max_iterations):
            logger.debug(f"[{self.__class__.__name__}] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Call Claude
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens if self.max_tokens and self.max_tokens > 0 else 4096,
                "system": self.system,
                "messages": messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools
            
            try:
                response = self.client.messages.create(**kwargs)
            except anthropic.RateLimitError:
                logger.warning("Rate limited — waiting 60s...")
                time.sleep(60)
                response = self.client.messages.create(**kwargs)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                raise
            
            # Track tokens
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            
            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract text from response
                final_text = self._extract_text(response.content)
                logger.debug(f"[{self.__class__.__name__}] Done. Tokens: in={self.total_input_tokens} out={self.total_output_tokens}")
                return final_text
            
            elif response.stop_reason == "tool_use":
                # Execute tools and continue
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.debug(f"[{self.__class__.__name__}] Tool call: {block.name}({json.dumps(block.input)[:100]})")
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                
                # Append assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            
            else:
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break
        
        logger.warning(f"[{self.__class__.__name__}] Hit max iterations ({self.max_iterations})")
        return self._extract_text(response.content) if response else ""

    def _run_gemini_tool_loop(self, messages: list[dict]) -> str:
        """Gemini function-calling agentic loop (mirrors Anthropic tool loop)."""
        # Build initial Gemini contents from messages
        contents: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                role = "model"
            if role not in ("user", "model"):
                continue
            contents.append({"role": role, "parts": [{"text": str(m.get("content", ""))}]})

        for iteration in range(self.max_iterations):
            logger.debug("[%s] Gemini tool loop iteration %d/%d", self.__class__.__name__, iteration + 1, self.max_iterations)

            resp = self._gemini.chat(
                model=self.provider_model,
                system=self.system,
                messages=[],  # passed via gemini_contents
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=self.tools,
                gemini_contents=contents,
            )
            self.total_input_tokens += resp.input_tokens
            self.total_output_tokens += resp.output_tokens

            if resp.finish_reason == "stop" or not resp.tool_calls:
                return resp.text

            # Append model turn (function calls) to contents
            model_parts = [
                {"functionCall": {"name": tc["name"], "args": tc["args"]}}
                for tc in resp.tool_calls
            ]
            if resp.text:
                model_parts.insert(0, {"text": resp.text})
            contents.append({"role": "model", "parts": model_parts})

            # Execute tools, build functionResponse parts
            response_parts = []
            for tc in resp.tool_calls:
                logger.debug("[%s] Gemini tool: %s(%s)", self.__class__.__name__, tc["name"], str(tc["args"])[:80])
                try:
                    result = self._execute_tool(tc["name"], tc["args"])
                except Exception as e:
                    result = {"error": str(e)}
                response_parts.append({
                    "functionResponse": {
                        "name": tc["name"],
                        "response": {"result": json.dumps(result, default=str)},
                    }
                })
            contents.append({"role": "user", "parts": response_parts})

        logger.warning("[%s] Gemini tool loop hit max iterations", self.__class__.__name__)
        return resp.text if resp else ""

    def _run_simple_provider(self, messages: list[dict]) -> str:
        if self.provider_name == "groq":
            resp = self._groq.chat(
                model=self.provider_model,
                system=self.system,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        elif self.provider_name == "gemini":
            resp = self._gemini.chat(
                model=self.provider_model,
                system=self.system,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        elif self.provider_name == "mistral":
            resp = self._mistral.chat(
                model=self.provider_model,
                system=self.system,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        elif self.provider_name == "openai":
            resp = self._openai.chat(
                model=self.provider_model,
                system=self.system,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        else:
            raise RuntimeError(f"Unknown provider: {self.provider_name}")

        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        return resp.text

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Execute a tool call. Subclasses override this to provide tool implementations.
        Base class returns an error for unknown tools.
        """
        return {"error": f"Tool '{tool_name}' not implemented in {self.__class__.__name__}"}

    def _extract_text(self, content_blocks) -> str:
        """Extract combined text from a list of content blocks."""
        texts = []
        for block in content_blocks:
            if hasattr(block, "type") and block.type == "text":
                texts.append(block.text)
        return "\n".join(texts)

    def get_token_usage(self) -> dict:
        """Return token usage summary."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        """Estimate API cost based on model pricing (approximate)."""
        # Provider-aware, approximate pricing per million tokens (estimation only).
        if self.provider_name == "anthropic":
            pricing = {
                "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
                "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
                "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
                "claude-opus-4-5": {"input": 5.00, "output": 25.00},
            }
            rates = pricing.get(self.model, {"input": 3.00, "output": 15.00})
        elif self.provider_name == "gemini":
            rates = {"input": getattr(config, "GEMINI_INPUT_PER_M", 0.0), "output": getattr(config, "GEMINI_OUTPUT_PER_M", 0.0)}
        elif self.provider_name == "groq":
            rates = {"input": getattr(config, "GROQ_INPUT_PER_M", 0.0), "output": getattr(config, "GROQ_OUTPUT_PER_M", 0.0)}
        elif self.provider_name == "mistral":
            rates = {"input": getattr(config, "MISTRAL_INPUT_PER_M", 0.0), "output": getattr(config, "MISTRAL_OUTPUT_PER_M", 0.0)}
        elif self.provider_name == "openai":
            # Unknown pricing for arbitrary OpenAI-compatible endpoints; default 0 unless user configures.
            rates = {"input": 0.0, "output": 0.0}
        else:
            rates = {"input": 0.0, "output": 0.0}
        cost = (self.total_input_tokens / 1_000_000) * rates["input"]
        cost += (self.total_output_tokens / 1_000_000) * rates["output"]
        return round(cost, 4)


class ResearchAgent(BaseAgent):
    """
    Extended base agent with built-in data-fetching tools.
    Used by bull/bear/scenario agents that need to query stock data and news.
    """

    def __init__(self, system_prompt: str, model: str = None, **kwargs):
        # Define standard research tools
        tools = [
            {
                "name": "get_stock_fundamentals",
                "description": "Get fundamental financial data for an Indian NSE stock. Returns PE, ROE, revenue growth, debt ratio, price data, analyst targets, and more.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "NSE stock symbol (e.g., RELIANCE, TCS, HDFCBANK). Do NOT include .NS suffix.",
                        }
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_recent_news",
                "description": "Get recent news articles (last 7 days) for an Indian stock. Returns article titles, summaries, dates, and sources. Use this to find recent catalysts, risks, and market sentiment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "NSE stock symbol (e.g., RELIANCE, TCS)",
                        },
                        "company_name": {
                            "type": "string",
                            "description": "Full company name for better news matching (e.g., Reliance Industries)",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Lookback window in days. Use 7 for research and 1-2 for thesis monitoring.",
                        },
                    },
                    "required": ["symbol", "company_name"],
                },
            },
            {
                "name": "get_macro_context",
                "description": "Get current Indian macro environment: Nifty 50 performance, USD/INR rate, crude oil price, India VIX, and sector index returns for the past month.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]
        
        super().__init__(system_prompt, tools=tools, model=model, **kwargs)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Map tool calls to data fetcher functions."""
        # Import here to avoid circular imports
        from data.fetcher import get_fundamentals, get_news_for_stock, get_macro_context
        
        if tool_name == "get_stock_fundamentals":
            symbol = tool_input["symbol"].strip().upper()
            return get_fundamentals(f"{symbol}.NS")
        
        elif tool_name == "get_recent_news":
            symbol = tool_input["symbol"].strip().upper()
            company_name = tool_input.get("company_name", symbol)
            days = int(tool_input.get("days", 7) or 7)
            return get_news_for_stock(symbol, company_name, days=max(1, days))
        
        elif tool_name == "get_macro_context":
            return get_macro_context()
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
