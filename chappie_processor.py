# chappie_processor.py

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from chappie_utils import parse_srt, seconds_to_time
import os

class ChappieProcessor:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name="gpt-3.5-turbo-16k",
            temperature=0.7,
            max_tokens=1000
        )

    def process_srt(self, srt_content: str) -> Dict[str, any]:
        entries = parse_srt(srt_content)
        chapters = self._generate_chapters(entries)
        chapter_summaries = self._generate_summaries(chapters)
        overall_summary = self._generate_overall_summary(srt_content)
        chapter_titles = self.generate_chapter_titles(chapters)

        for i, title in enumerate(chapter_titles):
            chapters[i]['title'] = title

        return {
            'chapters': chapters,
            'chapter_summaries': chapter_summaries,
            'overall_summary': overall_summary
        }

    def _generate_chapters(self, entries: List[Dict[str, any]]) -> List[Dict[str, any]]:
        chapters = []
        current_chapter_start = entries[0]['start']
        current_chapter_text = ""

        for i, entry in enumerate(entries):
            if i > 0 and i % 10 == 0:  # Create a new chapter every 10 entries
                chapters.append({
                    'start': current_chapter_start,
                    'end': entries[i-1]['end'],
                    'text': current_chapter_text.strip(),
                    'title': f"Chapter {len(chapters) + 1}"  # Add a default title
                })
                current_chapter_start = entry['start']
                current_chapter_text = ""
            
            current_chapter_text += entry['text'] + " "

        # Add the last chapter
        chapters.append({
            'start': current_chapter_start,
            'end': entries[-1]['end'],
            'text': current_chapter_text.strip(),
            'title': f"Chapter {len(chapters) + 1}"  # Add a default title
        })

        return chapters

    def _generate_summaries(self, chapters: List[Dict[str, any]]) -> List[str]:
        prompt = PromptTemplate(
            input_variables=["chapter_content"],
            template="Summarize the following chapter in one sentence: {chapter_content}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        summaries = []
        for chapter in chapters:
            summary = chain.invoke({"chapter_content": chapter['text'][:4000]})  # Limit to 4000 characters
            summaries.append(summary['text'])
        
        return summaries

    def _generate_overall_summary(self, srt_content: str) -> str:
        prompt = PromptTemplate(
            input_variables=["transcript"],
            template="Provide a brief summary of the following transcript: {transcript}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = chain.invoke({"transcript": srt_content[:4000]})  # Limit to 4000 characters
        return result['text']

    def generate_chapter_titles(self, chapters: List[Dict[str, any]]) -> List[str]:
        prompt = PromptTemplate(
            input_variables=["chapter_content"],
            template="Generate a short, descriptive title for the following chapter content: {chapter_content}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        titles = []
        for chapter in chapters:
            result = chain.invoke({"chapter_content": chapter['text'][:1000]})  # Limit to 1000 characters
            titles.append(result['text'].strip())
        
        return titles

    def process_directory(self, directory_path: str) -> Dict[str, Dict[str, any]]:
        results = {}
        for filename in os.listdir(directory_path):
            if filename.endswith(".srt"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    srt_content = file.read()
                results[filename] = self.process_srt(srt_content)
        return results