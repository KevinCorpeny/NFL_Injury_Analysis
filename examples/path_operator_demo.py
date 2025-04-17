class SimplePath:
    def __init__(self, path):
        # Store path as a string, removing any trailing slashes
        self.path = path.rstrip('/')
    
    def __truediv__(self, other):
        """Implement the '/' operator for path joining."""
        # Convert the other object to string if it isn't already
        other_str = str(other)
        
        # Join the paths with a forward slash
        new_path = f"{self.path}/{other_str}"
        
        # Return a new SimplePath object
        return SimplePath(new_path)
    
    def __str__(self):
        """String representation of the path."""
        return self.path

# Example usage
if __name__ == "__main__":
    # Create a base path
    base = SimplePath("/home/user")
    
    # Use the / operator to add more path components
    project_path = base / "projects" / "my_project"
    
    # Print the result
    print(f"Base path: {base}")
    print(f"Project path: {project_path}")
    
    # Show that it works with multiple components
    data_path = project_path / "data" / "raw"
    print(f"Data path: {data_path}") 