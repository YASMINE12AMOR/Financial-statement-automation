from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from langchainmemory import LangchainFlaskMemory, ConversationFlaskMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain