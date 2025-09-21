#!/usr/bin/env python3
"""
Script to convert a text file into a 2D array/list.
First dimension: breaks at newlines (\n)
Second dimension: breaks at spaces (' ')
"""

def text_to_2d_array(file_path):
    """
    Convert a text file into a 2D array/list.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: 2D list where each inner list represents a line split by spaces
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split by newlines to get lines
        lines = content.split('\n')
        
        # Split each line by spaces to get words
        result = [line.split(' ') for line in lines]
        
        return result
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def main():
    """Main function to demonstrate usage."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python text_to_2d_array.py <file_path>")
        print("Example: python text_to_2d_array.py sample.txt")
        return
    
    file_path = sys.argv[1]
    
    # Convert text file to 2D array
    result = text_to_2d_array(file_path)
    
    if result:
        print(f"Successfully converted '{file_path}' to 2D array.")
        print(f"Number of lines: {len(result)}")
        
        # Display first few lines as example
        print("\nFirst 3 lines:")
        for i, line in enumerate(result[79:82]):
            print(f"Line {i+1}: {line}")

if __name__ == "__main__":
    main()
