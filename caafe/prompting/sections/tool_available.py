def tool_use_section():
    return """
## Tools Available

In this environment you have access to a set of tools you can use to answer the user's question.

You can invoke functions by writing a "<actual_tool_name>" block like the following as part of your reply to the user:
<actual_tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
...
</actual_tool_name>

String and scalar parameters should be specified as is, while lists and objects should use JSON format.

Here are the available tools and their capabilities:
"""
