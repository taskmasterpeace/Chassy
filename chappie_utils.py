# chappie_utils.py

def parse_srt(srt_content: str) -> list:
    """
    Parse SRT content into a list of entry dictionaries.
    
    :param srt_content: Content of the SRT file
    :return: List of entry dictionaries
    """
    entries = []
    lines = srt_content.strip().split('\n')
    
    # Skip metadata
    start_index = 0
    for i, line in enumerate(lines):
        if line.isdigit():
            start_index = i
            break
    
    # Process SRT entries
    current_entry = {}
    for line in lines[start_index:]:
        if line.strip().isdigit():
            if current_entry:
                entries.append(current_entry)
                current_entry = {}
        elif ' --> ' in line:
            start, end = line.split(' --> ')
            current_entry['start'] = time_to_seconds(start.strip())
            current_entry['end'] = time_to_seconds(end.strip())
        elif line.strip():
            if 'text' not in current_entry:
                current_entry['text'] = line.strip()
            else:
                current_entry['text'] += ' ' + line.strip()
    
    if current_entry:
        entries.append(current_entry)
    
    return entries

def time_to_seconds(time_str: str) -> float:
    try:
        if ',' in time_str:
            time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError(f"Unexpected time format: {time_str}")
    except Exception as e:
        logging.error(f"Error parsing time: {time_str}")
        raise

def seconds_to_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"