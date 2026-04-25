const { ToolMessage } = require('@langchain/core/messages');
const { EModelEndpoint, ContentTypes } = require('librechat-data-provider');
const { HumanMessage, AIMessage, SystemMessage } = require('@langchain/core/messages');

/**
 * Formats a message to OpenAI Vision API payload format.
 *
 * @param {Object} params - The parameters for formatting.
 * @param {Object} params.message - The message object to format.
 * @param {string} [params.message.role] - The role of the message sender (must be 'user').
 * @param {string} [params.message.content] - The text content of the message.
 * @param {EModelEndpoint} [params.endpoint] - Identifier for specific endpoint handling
 * @param {Array<string>} [params.image_urls] - The image_urls to attach to the message.
 * @returns {(Object)} - The formatted message.
 */
const formatVisionMessage = ({ message, image_urls, endpoint }) => {
  if (endpoint === EModelEndpoint.anthropic) {
    message.content = [...image_urls, { type: ContentTypes.TEXT, text: message.content }];
    return message;
  }

  message.content = [{ type: ContentTypes.TEXT, text: message.content }, ...image_urls];

  return message;
};

/**
 * Formats a message to OpenAI payload format based on the provided options.
 *
 * @param {Object} params - The parameters for formatting.
 * @param {Object} params.message - The message object to format.
 * @param {string} [params.message.role] - The role of the message sender (e.g., 'user', 'assistant').
 * @param {string} [params.message._name] - The name associated with the message.
 * @param {string} [params.message.sender] - The sender of the message.
 * @param {string} [params.message.text] - The text content of the message.
 * @param {string} [params.message.content] - The content of the message.
 * @param {Array<string>} [params.message.image_urls] - The image_urls attached to the message for Vision API.
 * @param {string} [params.userName] - The name of the user.
 * @param {string} [params.assistantName] - The name of the assistant.
 * @param {string} [params.endpoint] - Identifier for specific endpoint handling
 * @param {boolean} [params.langChain=false] - Whether to return a LangChain message object.
 * @returns {(Object|HumanMessage|AIMessage|SystemMessage)} - The formatted message.
 */
const formatMessage = ({ message, userName, assistantName, endpoint, langChain = false }) => {
  let { role: _role, _name, sender, text, content: _content, lc_id } = message;
  if (lc_id && lc_id[2] && !langChain) {
    const roleMapping = {
      SystemMessage: 'system',
      HumanMessage: 'user',
      AIMessage: 'assistant',
    };
    _role = roleMapping[lc_id[2]];
  }
  const role = _role ?? (sender && sender?.toLowerCase() === 'user' ? 'user' : 'assistant');
  const content = _content ?? text ?? '';
  const formattedMessage = {
    role,
    content,
  };

  const { image_urls } = message;
  if (Array.isArray(image_urls) && image_urls.length > 0 && role === 'user') {
    return formatVisionMessage({
      message: formattedMessage,
      image_urls: message.image_urls,
      endpoint,
    });
  }

  if (_name) {
    formattedMessage.name = _name;
  }

  if (userName && formattedMessage.role === 'user') {
    formattedMessage.name = userName;
  }

  if (assistantName && formattedMessage.role === 'assistant') {
    formattedMessage.name = assistantName;
  }

  if (formattedMessage.name) {
    // Conform to API regex: ^[a-zA-Z0-9_-]{1,64}$
    // https://community.openai.com/t/the-format-of-the-name-field-in-the-documentation-is-incorrect/175684/2
    formattedMessage.name = formattedMessage.name.replace(/[^a-zA-Z0-9_-]/g, '_');

    if (formattedMessage.name.length > 64) {
      formattedMessage.name = formattedMessage.name.substring(0, 64);
    }
  }

  if (!langChain) {
    return formattedMessage;
  }

  if (role === 'user') {
    return new HumanMessage(formattedMessage);
  } else if (role === 'assistant') {
    return new AIMessage(formattedMessage);
  } else {
    return new SystemMessage(formattedMessage);
  }
};

/**
 * Formats an array of messages for LangChain.
 *
 * @param {Array<Object>} messages - The array of messages to format.
 * @param {Object} formatOptions - The options for formatting each message.
 * @param {string} [formatOptions.userName] - The name of the user.
 * @param {string} [formatOptions.assistantName] - The name of the assistant.
 * @returns {Array<(HumanMessage|AIMessage|SystemMessage)>} - The array of formatted LangChain messages.
 */
const formatLangChainMessages = (messages, formatOptions) =>
  messages.map((msg) => formatMessage({ ...formatOptions, message: msg, langChain: true }));

/**
 * Formats a LangChain message object by merging properties from `lc_kwargs` or `kwargs` and `additional_kwargs`.
 *
 * @param {Object} message - The message object to format.
 * @param {Object} [message.lc_kwargs] - Contains properties to be merged. Either this or `message.kwargs` should be provided.
 * @param {Object} [message.kwargs] - Contains properties to be merged. Either this or `message.lc_kwargs` should be provided.
 * @param {Object} [message.kwargs.additional_kwargs] - Additional properties to be merged.
 *
 * @returns {Object} The formatted LangChain message.
 */
const formatFromLangChain = (message) => {
  const { additional_kwargs, ...message_kwargs } = message.lc_kwargs ?? message.kwargs;
  return {
    ...message_kwargs,
    ...additional_kwargs,
  };
};

/**
 * Formats an array of messages for LangChain, handling tool calls and creating ToolMessage instances.
 *
 * @param {Array<Partial<TMessage>>} payload - The array of messages to format.
 * @returns {Array<(HumanMessage|AIMessage|SystemMessage|ToolMessage)>} - The array of formatted LangChain messages, including ToolMessages for tool calls.
 */
const formatAgentMessages = (payload) => {
  const messages = [];

  for (const message of payload) {
    if (typeof message.content === 'string') {
      message.content = [{ type: ContentTypes.TEXT, [ContentTypes.TEXT]: message.content }];
    }
    if (message.role !== 'assistant') {
      messages.push(formatMessage({ message, langChain: true }));
      continue;
    }

    let currentContent = [];
    let lastAIMessage = null;

    let hasReasoning = false;
    for (const part of message.content) {
      if (part.type === ContentTypes.TEXT && part.tool_call_ids) {
        /*
        If there's pending content, it needs to be aggregated as a single string to prepare for tool calls.
        For Anthropic models, the "tool_calls" field on a message is only respected if content is a string.
         */
        if (currentContent.length > 0) {
          let content = currentContent.reduce((acc, curr) => {
            if (curr.type === ContentTypes.TEXT) {
              return `${acc}${curr[ContentTypes.TEXT]}\n`;
            }
            return acc;
          }, '');
          content = `${content}\n${part[ContentTypes.TEXT] ?? ''}`.trim();
          lastAIMessage = new AIMessage({ content });
          messages.push(lastAIMessage);
          currentContent = [];
          continue;
        }

        // Create a new AIMessage with this text and prepare for tool calls
        lastAIMessage = new AIMessage({
          content: part.text || '',
        });

        messages.push(lastAIMessage);
      } else if (part.type === ContentTypes.TOOL_CALL) {
        if (!lastAIMessage) {
          throw new Error('Invalid tool call structure: No preceding AIMessage with tool_call_ids');
        }

        // Note: `tool_calls` list is defined when constructed by `AIMessage` class, and outputs should be excluded from it
        const { output, args: _args, ...tool_call } = part.tool_call;
        // TODO: investigate; args as dictionary may need to be provider-or-tool-specific
        let args = _args;
        try {
          args = JSON.parse(_args);
        } catch (e) {
          if (typeof _args === 'string') {
            args = { input: _args };
          }
        }

        tool_call.args = args;
        lastAIMessage.tool_calls.push(tool_call);

        // Add the corresponding ToolMessage
        messages.push(
          new ToolMessage({
            tool_call_id: tool_call.id,
            name: tool_call.name,
            content: output || '',
          }),
        );
      } else if (part.type === ContentTypes.THINK) {
        hasReasoning = true;
        continue;
      } else if (part.type === ContentTypes.ERROR || part.type === ContentTypes.AGENT_UPDATE) {
        continue;
      } else {
        currentContent.push(part);
      }
    }

    if (hasReasoning) {
      currentContent = currentContent
        .reduce((acc, curr) => {
          if (curr.type === ContentTypes.TEXT) {
            return `${acc}${curr[ContentTypes.TEXT]}\n`;
          }
          return acc;
        }, '')
        .trim();
    }

    if (currentContent.length > 0) {
      messages.push(new AIMessage({ content: currentContent }));
    }
  }

  return messages;
};

/**
 * Extracts reasoning content string from THINK parts in a message's content array.
 * @param {Array} contentParts - Content parts array from a message
 * @returns {string|null} - Concatenated reasoning text, or null if no reasoning present
 */
function extractReasoningContent(contentParts) {
  if (!Array.isArray(contentParts)) {
    return null;
  }
  const thinkParts = contentParts.filter(
    (part) => part && (part.type === ContentTypes.THINK || part.type === 'think'),
  );
  if (thinkParts.length === 0) {
    return null;
  }
  return (
    thinkParts
      .map((part) => part.think || part[ContentTypes.THINK] || '')
      .join('\n')
      .trim() || null
  );
}

/**
 * Adds `reasoning_content` to `additional_kwargs` on formatted AIMessages that
 * correspond to payload assistant messages with THINK content parts.
 *
 * DeepSeek (and potentially other providers) require that `reasoning_content`
 * from previous assistant responses be passed back to the API in subsequent
 * multi-turn conversations when thinking mode is enabled.
 *
 * @param {Array<Object>} payload - Original message payload before formatting
 * @param {Array<BaseMessage>} formattedMessages - Messages returned by formatAgentMessages
 * @returns {Array<BaseMessage>} - The formatted messages, mutated in place
 */
function addReasoningContentToMessages(payload, formattedMessages) {
  // Build an index-based lookup of reasoning content per payload message
  const reasoningByPayloadIndex = payload.map((msg) => {
    if (!msg || msg.role !== 'assistant') {
      return null;
    }
    return extractReasoningContent(msg.content);
  });

  if (reasoningByPayloadIndex.every((r) => !r)) {
    return formattedMessages;
  }

  // Walk through payload and formatted messages in parallel.
  // Each non-assistant payload message maps to exactly one formatted message.
  // Each assistant payload message maps to a sequence of AIMessage(s) + ToolMessage(s).
  // We track groups by knowing that formatAssistantMessage produces:
  //   - AIMessage for each TEXT+tool_call_ids part
  //   - ToolMessage for each TOOL_CALL part
  //   - AIMessage for final TEXT content parts
  let payloadIdx = 0;
  let formattedIdx = 0;

  while (payloadIdx < payload.length && formattedIdx < formattedMessages.length) {
    const payloadMsg = payload[payloadIdx];
    const reasoning = reasoningByPayloadIndex[payloadIdx];
    const fmsg = formattedMessages[formattedIdx];
    const msgType = fmsg && typeof fmsg._getType === 'function' ? fmsg._getType() : null;

    if (!payloadMsg || payloadMsg.role !== 'assistant') {
      // Non-assistant: 1 payload message → 1 formatted message
      payloadIdx++;
      formattedIdx++;
      continue;
    }

    // Assistant message: count how many formatted messages it produced
    // by analyzing the content structure.
    let aimessageCount = 0;
    let toolMessageCount = 0;
    if (Array.isArray(payloadMsg.content)) {
      let lastToolCall = false;
      for (const part of payloadMsg.content) {
        if (!part) {
          continue;
        }
        if (part.type === ContentTypes.TEXT && part.tool_call_ids) {
          aimessageCount++;
          lastToolCall = true;
        } else if (part.type === ContentTypes.TOOL_CALL) {
          toolMessageCount++;
          lastToolCall = true;
        } else if (part.type === ContentTypes.THINK) {
          // THINK is skipped, does not produce a formatted message
        } else if (part.type === ContentTypes.ERROR || part.type === ContentTypes.AGENT_UPDATE) {
          // ERROR/AGENT_UPDATE skipped
        } else {
          lastToolCall = false;
        }
      }
      // If there are accumulated TEXT parts without tool_call_ids, they produce one AIMessage
      const hasFinalTextContent = payloadMsg.content.some(
        (part) =>
          part &&
          part.type === ContentTypes.TEXT &&
          !part.tool_call_ids,
      );
      if (hasFinalTextContent) {
        aimessageCount++;
      }
    }

    // Add reasoning_content to all AIMessages in this group
    const totalFormattedForMsg = aimessageCount + toolMessageCount;
    for (let i = 0; i < totalFormattedForMsg && formattedIdx < formattedMessages.length; i++) {
      const current = formattedMessages[formattedIdx];
      const currentType = current && typeof current._getType === 'function' ? current._getType() : null;
      if (reasoning && currentType === 'ai') {
        if (!current.additional_kwargs) {
          current.additional_kwargs = {};
        }
        current.additional_kwargs.reasoning_content = reasoning;
      }
      formattedIdx++;
    }

    payloadIdx++;
  }

  return formattedMessages;
}

module.exports = {
  formatMessage,
  formatFromLangChain,
  formatAgentMessages,
  formatLangChainMessages,
  extractReasoningContent,
  addReasoningContentToMessages,
};
