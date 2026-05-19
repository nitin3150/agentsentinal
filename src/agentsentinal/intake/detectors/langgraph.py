import inspect
from agentsentinal.models import AgentProfile
from typing import Optional
from typing import Any

SKIP_NODES=['__start__','__end__']

class LangGraphDetector():
    MODEL_NODE_NAMES=('model','agent')

    def __init__(self,agent):
        self.agent = agent
        # self.graph = None
    
    def can_handle(self) -> bool:
        return hasattr(self.agent,'nodes')
    
    def _extract_system_prompt(self, model_node:Any) -> Optional[str]:
        bound = getattr(model_node,"bound",None) or model_node
        fn = getattr(bound,"afunc",None) or getattr(bound,"func",None) or bound
        
        if not callable(fn):
            return None
        
        try:
            nonlocals = inspect.getclosurevars(fn).nonlocals
        except Exception:
            nonlocals = {}
        
        for name in ('system_prompt','system_message','prompt','state_modifiers'):
            val = nonlocals.get(name)
            if val is None:
                continue
            
            if isinstance(val,str) and len(val)>10:
                return val
            
            if hasattr(val,'content') and isinstance(val.content,str):
                return val.content
        return None

    def _convert_tool(self,tool):
        schema = {}
        args_schema = getattr(tool,"arg_schema",None)
        if args_schema:
            try:
                schema = args_schema.model_json_schema()
            except Exception:
                pass
        
        return {
            "name":getattr(tool,'name',None),
            "description":getattr(tool,'description',None),
            "parameters":schema
        }

    def _extract_tool_definitions(self, tools_node:Any) -> list[Any]:
        bound = getattr(tools_node,'bound',None) or tools_node
        for attr in ('tools_by_name','_tools_by_name'):
            registry = getattr(bound,attr,None)
            if isinstance(registry,dict) and registry:
                return [self._convert_tool(t) for t in registry.values()]
        
        data = getattr(tools_node, "data", None)
        for attr in ("tools_by_name", "_tools_by_name"):
            registry = getattr(data, attr, None)
            if isinstance(registry, dict) and registry:
                return [self._convert_tool(t) for t in registry.values()]

        tools_attr = getattr(bound, "tools", None) or getattr(data, "tools", None)
        if isinstance(tools_attr, (list, tuple)):
            return list(tools_attr)
        return []
            

    def __call__(self) -> AgentProfile:
        result = AgentProfile(
            system_prompt    = "",
            framework        = "langgraph",
            tool_definitions = [],
            warnings         = [],
        )

        # Get nodes
        try:
            nodes = dict(self.agent.nodes)
        except Exception:
            result.warnings.append("Could not read graph nodes")
            return result

        # Find model node
        model_node = next(
            (nodes[name] for name in self.MODEL_NODE_NAMES if name in nodes),
            None
        )
        if model_node is None:
            result.warnings.append("Model node not found")
        else:
            prompt = self._extract_system_prompt(model_node)
            if prompt:
                result.system_prompt = prompt
            else:
                result.warnings.append("System prompt not found in closure")

        # Find tools node
        tools_node = nodes.get("tools")
        if tools_node is not None:
            tools = self._extract_tool_definitions(tools_node)
            if tools:
                result.tool_definitions = tools
            else:
                result.warnings.append("No tools found")

        return result