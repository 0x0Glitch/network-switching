import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import pytest
from decimal import Decimal

# Add parent directory to path to import chatbot module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
import chatbot
from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    CdpWalletProviderConfig,
    skywire_action_provider,
)
from coinbase_agentkit.action_providers.skywire.skywire_action_provider import SkywireActionProvider
from coinbase_agentkit_langchain import get_langchain_tools
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


class TestChatbot(unittest.TestCase):
    """Test suite for the CDP chatbot."""

    @patch('chatbot.ChatOpenAI')
    @patch('chatbot.get_langchain_tools')
    @patch('chatbot.create_react_agent')
    @patch('chatbot.CdpWalletProvider')
    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_initialize_agent(self, mock_file, mock_wallet_provider, mock_react_agent, 
                            mock_get_tools, mock_chat_openai):
        """Test the agent initialization function."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_wallet = MagicMock()
        mock_wallet.export_wallet.return_value.to_dict.return_value = {"key": "updated_value"}
        mock_wallet_provider.return_value = mock_wallet
        
        mock_tools = [MagicMock()]
        mock_get_tools.return_value = mock_tools
        
        mock_agent = MagicMock()
        mock_react_agent.return_value = (mock_agent, {"config": "test"})
        
        # Call the function
        agent, config = chatbot.initialize_agent()
        
        # Assertions
        mock_chat_openai.assert_called_once_with(model="gpt-4o")
        mock_wallet_provider.assert_called_once()
        mock_get_tools.assert_called_once()
        mock_react_agent.assert_called_once()
        mock_file().write.assert_called_once_with(json.dumps({"key": "updated_value"}))
        
        self.assertEqual(agent, mock_agent)
        self.assertEqual(config, {"config": "test"})

    @patch('chatbot.time')
    @patch('chatbot.sys.exit')
    def test_run_autonomous_mode_keyboard_interrupt(self, mock_exit, mock_time):
        """Test autonomous mode with keyboard interrupt."""
        mock_agent = MagicMock()
        mock_agent.stream.side_effect = KeyboardInterrupt()
        mock_config = {"test": "config"}
        
        with patch('builtins.print') as mock_print:
            chatbot.run_autonomous_mode(mock_agent, mock_config)
            
            mock_exit.assert_called_once_with(0)
            mock_print.assert_any_call("Goodbye Agent!")

    @patch('chatbot.time.time')
    @patch('chatbot.time.sleep')
    def test_run_autonomous_mode_normal(self, mock_sleep, mock_time):
        """Test autonomous mode normal execution."""
        mock_agent = MagicMock()
        # Mock stream to return chunks that simulate agent and tool responses
        mock_agent.stream.return_value = [
            {"agent": {"messages": [MagicMock(content="Agent response")]}},
            {"tools": {"messages": [MagicMock(content="Tool response")]}}
        ]
        mock_config = {"test": "config"}
        
        # Set time to return a specific value so we can test a specific thought option
        mock_time.return_value = 0  # This will select the first thought option
        
        # Setup to run once then raise KeyboardInterrupt to exit the loop
        mock_sleep.side_effect = [None, KeyboardInterrupt()]
        
        with patch('builtins.print') as mock_print:
            try:
                chatbot.run_autonomous_mode(mock_agent, mock_config)
            except KeyboardInterrupt:
                pass
            
            # Check that correct prints were made
            mock_print.assert_any_call("Starting autonomous mode...")
            mock_print.assert_any_call("Agent response")
            mock_print.assert_any_call("Tool response")
            mock_print.assert_any_call("-------------------")
            
            # Verify agent.stream was called with the expected thought
            expected_thought = "Demonstrate bridging 0.2 sETH from Base Sepolia to Optimism Sepolia: crosschain_burn on the source, crosschain_mint on the destination, then confirm final sETH balance for the user."
            mock_agent.stream.assert_called_with(
                {"messages": [HumanMessage(content=expected_thought)]}, mock_config
            )

    @patch('chatbot.input')
    @patch('chatbot.sys.exit')
    def test_run_chat_mode_exit(self, mock_exit, mock_input):
        """Test chat mode with exit command."""
        mock_agent = MagicMock()
        mock_config = {"test": "config"}
        mock_input.side_effect = ["exit"]
        
        with patch('builtins.print') as mock_print:
            chatbot.run_chat_mode(mock_agent, mock_config)
            
            mock_print.assert_any_call("Starting chat mode... Type 'exit' to end.")
            # Verify agent.stream was not called because we exited
            mock_agent.stream.assert_not_called()

    @patch('chatbot.input')
    def test_run_chat_mode_with_input(self, mock_input):
        """Test chat mode with user input."""
        mock_agent = MagicMock()
        # Mock stream to return chunks that simulate agent and tool responses
        mock_agent.stream.return_value = [
            {"agent": {"messages": [MagicMock(content="Agent response")]}},
            {"tools": {"messages": [MagicMock(content="Tool response")]}}
        ]
        mock_config = {"test": "config"}
        
        # Simulate user input followed by exit
        mock_input.side_effect = ["Hello, agent!", "exit"]
        
        with patch('builtins.print') as mock_print:
            chatbot.run_chat_mode(mock_agent, mock_config)
            
            # Check proper prints
            mock_print.assert_any_call("Starting chat mode... Type 'exit' to end.")
            mock_print.assert_any_call("Agent response")
            mock_print.assert_any_call("Tool response")
            
            # Verify agent.stream was called with user input
            mock_agent.stream.assert_called_once_with(
                {"messages": [HumanMessage(content="Hello, agent!")]}, mock_config
            )

    @patch('chatbot.input')
    @patch('chatbot.sys.exit')
    def test_run_chat_mode_keyboard_interrupt(self, mock_exit, mock_input):
        """Test chat mode with keyboard interrupt."""
        mock_agent = MagicMock()
        mock_config = {"test": "config"}
        mock_input.side_effect = KeyboardInterrupt()
        
        with patch('builtins.print') as mock_print:
            chatbot.run_chat_mode(mock_agent, mock_config)
            
            mock_exit.assert_called_once_with(0)
            mock_print.assert_any_call("Goodbye Agent!")

    @patch('chatbot.input')
    def test_choose_mode_chat(self, mock_input):
        """Test mode selection with chat."""
        mock_input.return_value = "chat"
        
        with patch('builtins.print'):
            result = chatbot.choose_mode()
            self.assertEqual(result, "chat")

    @patch('chatbot.input')
    def test_choose_mode_auto(self, mock_input):
        """Test mode selection with auto."""
        mock_input.return_value = "auto"
        
        with patch('builtins.print'):
            result = chatbot.choose_mode()
            self.assertEqual(result, "auto")

    @patch('chatbot.input')
    def test_choose_mode_number_inputs(self, mock_input):
        """Test mode selection with number inputs."""
        mock_input.side_effect = ["1", "2"]
        
        with patch('builtins.print'):
            result1 = chatbot.choose_mode()
            self.assertEqual(result1, "chat")
            
            result2 = chatbot.choose_mode()
            self.assertEqual(result2, "auto")

    @patch('chatbot.input')
    def test_choose_mode_invalid_then_valid(self, mock_input):
        """Test mode selection with invalid input followed by valid input."""
        mock_input.side_effect = ["invalid", "chat"]
        
        with patch('builtins.print') as mock_print:
            result = chatbot.choose_mode()
            self.assertEqual(result, "chat")
            mock_print.assert_any_call("Invalid choice. Please try again.")

    @patch('chatbot.choose_mode')
    @patch('chatbot.initialize_agent')
    @patch('chatbot.run_chat_mode')
    def test_main_chat_mode(self, mock_run_chat, mock_init_agent, mock_choose_mode):
        """Test main function with chat mode."""
        mock_choose_mode.return_value = "chat"
        mock_agent = MagicMock()
        mock_config = {"test": "config"}
        mock_init_agent.return_value = (mock_agent, mock_config)
        
        chatbot.main()
        
        mock_init_agent.assert_called_once()
        mock_choose_mode.assert_called_once()
        mock_run_chat.assert_called_once_with(agent_executor=mock_agent, config=mock_config)

    @patch('chatbot.choose_mode')
    @patch('chatbot.initialize_agent')
    @patch('chatbot.run_autonomous_mode')
    def test_main_auto_mode(self, mock_run_auto, mock_init_agent, mock_choose_mode):
        """Test main function with auto mode."""
        mock_choose_mode.return_value = "auto"
        mock_agent = MagicMock()
        mock_config = {"test": "config"}
        mock_init_agent.return_value = (mock_agent, mock_config)
        
        chatbot.main()
        
        mock_init_agent.assert_called_once()
        mock_choose_mode.assert_called_once()
        mock_run_auto.assert_called_once_with(agent_executor=mock_agent, config=mock_config)


class TestSkywireIntegration(unittest.TestCase):
    """Test the integration with Skywire, focusing on the issues from error logs."""
    
    @patch('coinbase_agentkit.CdpWalletProvider')
    def test_skywire_argument_type_conversion(self, mock_wallet_provider):
        """Test that SkywireActionProvider properly converts string amount to uint256."""
        # Create a SkywireProvider instance
        skywire_provider = SkywireActionProvider()
        
        # Mock wallet provider
        mock_wallet = MagicMock()
        mock_wallet_provider.return_value = mock_wallet
        
        # Configure the mock to return expected values
        mock_wallet.web3 = MagicMock()
        
        # Test the _convert_amount_to_wei method with different inputs
        wei_amount = skywire_provider._convert_amount_to_wei("0.0000001")
        self.assertEqual(wei_amount, 100000000000)  # 0.0000001 ETH in wei
        
        # Test with other formats
        self.assertEqual(skywire_provider._convert_amount_to_wei("1"), 1000000000000000000)  # 1 ETH in wei
        self.assertEqual(skywire_provider._convert_amount_to_wei("0.1 sETH"), 100000000000000000)  # 0.1 ETH in wei
        
        # Test with very small amount
        self.assertEqual(skywire_provider._convert_amount_to_wei("0.000000000000000001"), 1)  # Smallest unit
    
    @patch('coinbase_agentkit.action_providers.skywire.skywire_action_provider.SmartContract')
    @patch('coinbase_agentkit.CdpWalletProvider')
    def test_skywire_crosschain_burn_proper_types(self, mock_wallet_provider, mock_smart_contract):
        """Test that crosschain_burn passes proper integer type to contract."""
        # Create a SkywireProvider instance
        skywire_provider = SkywireActionProvider()
        
        # Mock wallet provider and invoke_contract
        mock_wallet = MagicMock()
        mock_wallet_provider.return_value = mock_wallet
        
        # Setup invocation mock
        mock_invocation = MagicMock()
        mock_wallet.invoke_contract.return_value = mock_invocation
        
        # Configure mocks for get_address and get_chain_id
        mock_wallet.get_address.return_value = "0x888dc43F8aF62eafb2B542e309B836CA9683E410"
        mock_wallet.current_network_id = "base-sepolia"
        mock_wallet.get_chain_id.return_value = 84532
        
        # Mock balance check (user has enough balance)
        def mock_read(network, address, method, abi, *args):
            if method == "balanceOf":
                return 10**19  # User has 10 sETH
            return None
        
        mock_smart_contract.read.side_effect = mock_read
        
        # Call the method with string amount
        skywire_provider.crosschain_burn(mock_wallet, {
            "amount": "0.0000001",
            "from_address": "0x1234567890123456789012345678901234567890",
            "chain_id": 84532
        })
        
        # Verify invoke_contract was called with correct parameters
        mock_wallet.invoke_contract.assert_called_once()
        args, kwargs = mock_wallet.invoke_contract.call_args
        
        # Check that amount was properly converted to integer
        self.assertIsInstance(kwargs["args"]["amount"], int)
        self.assertEqual(kwargs["args"]["amount"], 100000000000)  # 0.0000001 ETH in wei
    
    @patch('coinbase_agentkit.action_providers.skywire.skywire_action_provider.SmartContract')
    @patch('coinbase_agentkit.CdpWalletProvider')
    def test_skywire_gas_estimation_error_handling(self, mock_wallet_provider, mock_smart_contract):
        """Test that gas estimation errors are handled gracefully."""
        # Create a SkywireProvider instance
        skywire_provider = SkywireActionProvider()
        
        # Mock wallet provider
        mock_wallet = MagicMock()
        mock_wallet_provider.return_value = mock_wallet
        
        # Configure mocks
        mock_wallet.get_address.return_value = "0x888dc43F8aF62eafb2B542e309B836CA9683E410"
        mock_wallet.current_network_id = "base-sepolia"
        mock_wallet.get_chain_id.return_value = 84532
        
        # Mock invoke_contract to raise AttributeError with 'estimateGas' in message
        mock_wallet.invoke_contract.side_effect = AttributeError("estimateGas failed for some reason")
        
        # Mock balance check (user has enough balance)
        mock_smart_contract.read.return_value = 10**19  # User has 10 sETH
        
        # Call the method
        result = skywire_provider.crosschain_burn(mock_wallet, {
            "amount": "0.0000001",
            "from_address": "0x1234567890123456789012345678901234567890",
            "chain_id": 84532
        })
        
        # Verify error was handled correctly
        self.assertIn("Failed to execute crosschainBurn", result)
        self.assertIn("estimateGas", result)
    
    @patch('coinbase_agentkit.action_providers.skywire.skywire_action_provider.SmartContract')
    @patch('coinbase_agentkit.CdpWalletProvider')
    def test_skywire_network_switching(self, mock_wallet_provider, mock_smart_contract):
        """Test that network switching works properly."""
        # Create a SkywireProvider instance
        skywire_provider = SkywireActionProvider()
        
        # Mock wallet provider
        mock_wallet = MagicMock()
        mock_wallet_provider.return_value = mock_wallet
        
        # Configure mocks
        mock_wallet.get_address.return_value = "0x888dc43F8aF62eafb2B542e309B836CA9683E410"
        mock_wallet.current_network_id = "optimism-sepolia"  # Starting on a different network
        mock_wallet.get_chain_id.return_value = 11155420
        
        # Mock balance check (user has enough balance)
        mock_smart_contract.read.return_value = 10**19  # User has 10 sETH
        
        # Configure switch_network to return True and change current_network_id
        def mock_switch_network(network_id):
            if network_id == "base-sepolia":
                mock_wallet.current_network_id = "base-sepolia"
                mock_wallet.get_chain_id.return_value = 84532
                return True
            return False
        
        mock_wallet.switch_network.side_effect = mock_switch_network
        mock_invocation = MagicMock()
        mock_wallet.invoke_contract.return_value = mock_invocation
        
        # Call the method with different chain than current
        skywire_provider.crosschain_burn(mock_wallet, {
            "amount": "0.0000001",
            "from_address": "0x1234567890123456789012345678901234567890",
            "chain_id": 84532  # Base Sepolia
        })
        
        # Verify switch_network was called
        mock_wallet.switch_network.assert_called_once_with("base-sepolia")

if __name__ == "__main__":
    unittest.main()
