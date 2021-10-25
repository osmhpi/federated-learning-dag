import collections
import json
import os
import random
import re
import sys

"""
Assumed text structure

Title

Additional information

Chapter
  Character.
    Here is the text of the character.
    And some more.
  (stage direction)
    Some more text after the stage direction.

  Different character [doing stuff].
    This is some more text.
  [stage directions can both be in normal bracktes as well as in square brackets].

  Third character (stage directions can both be in normal bracktes as well as in square brackets).
    Saying cool stuff.

  Last character (only following stage directions).

(Some more stage directions,
which are too long to fit in one line.)

Next Chapter

...
"""
# Regular expression to capture an actors name
CHARACTER_RE = re.compile(r'^  (?![\s\(\[])([a-zA-Z][a-zA-Z ]*)(?:\.| [\(\[][^\)\]]+[\)\]]\.)')
# Regular expression to capture content of an actor
CONT_RE = re.compile(r'^    (.*)')
# Regular expression to find the end of the ebook
EOB = re.compile(r'\*\*\* END OF EBOOK \*\*\*')

def _match_character_regex(line):
    return CHARACTER_RE.match(line)

def _match_continuation_regex(line):
    return CONT_RE.match(line)

def _match_end_of_ebook_regex(line):
    return EOB.match(line)

def _extract_character_data(content, title):
    """Removes unecessary data from play and extracts data per character."""
    # dict from character to list of lines
    character_data = {}
    # Track discarded lines.
    discarded_lines = []
    # Track, which users belong to this play
    users_in_play = {}

    slines = content.splitlines(True)[1:]

    current_character = None
    for line in slines:
        # Check, if we found the beginning of a new passage for an user
        match = _match_character_regex(line)
        if match:
            character = match.group(1)
            character = character.upper()
            character = play_and_character(title, character)
            current_character = _sanitized_umlaute(character)
            if current_character not in character_data:
                character_data[current_character] = []
            users_in_play[current_character] = title
        elif current_character:
            # Check, if we have content for the current character
            match = _match_continuation_regex(line)
            if match:
                character_data[current_character].append(match.group(1))
                continue
            # Check if we have reached the end of the ebook
            match = _match_end_of_ebook_regex(line)
            if match:
                break
    return character_data, users_in_play

def _remove_nonalphanumerics(filename):
    return re.sub('\\W+', '_', filename)

def play_and_character(play, character):
    return _remove_nonalphanumerics((play + '_' + character).replace(' ', '_'))

def _sanitized_umlaute(text):
        sanitized_ae = re.sub('\\Ä+', 'AE', text)
        sanitized_oe = re.sub('\\Ö+', 'OE', sanitized_ae)
        sanitized_ue = re.sub('\\Ü+', 'UE', sanitized_oe)
        return sanitized_ue

def _write_data_by_character(examples, output_directory):
    """Writes a collection of data files by play & character."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for character_name, sound_bites in examples.items():
        filename = os.path.join(output_directory, character_name + '.txt')
        filename = _sanitized_umlaute(filename)
        with open(filename, 'w') as output:
            for sound_bite in sound_bites:
                output.write(sound_bite + '\n')

def main(argv):
    print('Splitting each .txt between users')
    input_path_plays = argv[0]
    output_directory = argv[1]

    data_by_users = {}
    users_and_plays = {}

    for subdir, dirs, files in os.walk(input_path_plays):
        for filename in files:
            if filename.endswith(".txt"):
                print("Processing %s now." % filename)
                filepath = subdir + os.sep + filename

                with open(filepath, 'r') as input_file:
                    play = input_file.read()

                current_play = filename[:-4].upper()
                users, participations = _extract_character_data(play, current_play)
                data_by_users.update(users)
                users_and_plays.update(participations)
                
                _write_data_by_character(users, os.path.join(output_directory, 'by_play_and_character/'))

    with open(os.path.join(output_directory, 'users_and_plays.json'), 'w') as ouf:
        json.dump(users_and_plays, ouf)

if __name__ == '__main__':
    main(sys.argv[1:])
