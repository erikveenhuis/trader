---
description: General guidelines for maintaining code quality, readability, and collaboration.
globs: ["**/*.py", "**/*.md"]
---

# General Coding Guidelines

## Clarity and Readability
- Write code that is easy to understand. Prioritize clarity over conciseness if there's a trade-off.
- Use meaningful names for variables, functions, classes, and modules. Names should clearly indicate their purpose.
- Avoid overly complex expressions or one-liners that are hard to decipher.
- Break down complex logic into smaller, manageable functions or methods.

## Consistency
- Strive for a consistent coding style throughout the project.
- Follow established conventions for formatting, naming, and project structure.
- If modifying existing code, adapt to the style of that module or file.

## Simplicity (Keep It Simple, Stupid - KISS)
- Prefer simple solutions over complex ones.
- Avoid premature optimization. Optimize only when there's a proven performance bottleneck.
- Don't over-engineer solutions.

## Don't Repeat Yourself (DRY)
- Identify and eliminate redundant code by abstracting it into reusable functions, classes, or modules.
- Aim for a single source of truth for any piece of information or logic.

## Comments and Documentation
- Write comments to explain *why* code is written a certain way, especially for complex or non-obvious logic.
- Avoid commenting on *what* the code does if the code itself is clear.
- Keep comments up-to-date with code changes.
- Document public APIs (functions, classes, methods) with clear docstrings explaining their purpose, parameters, and return values.

## Error Handling
- Implement robust error handling. Anticipate potential failure points.
- Use exceptions appropriately for exceptional situations.
- Provide informative error messages.

## Testing
- Write unit tests for critical components and business logic.
- Aim for good test coverage.
- Tests should be readable, maintainable, and fast.

## Version Control (Git)
- Make small, atomic commits with clear and descriptive messages.
- Use feature branches for new development and bug fixes.
- Regularly pull changes from the main branch to avoid large merge conflicts.

## Code Reviews
- Participate in code reviews, both as a reviewer and an author.
- Provide constructive feedback. Focus on improving code quality and adherence to guidelines.
- Be open to receiving feedback and learning from it. 