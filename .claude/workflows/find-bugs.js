export const meta = {
  name: 'find-bugs',
  description: 'Find and list potential bugs across the codebase',
  phases: [
    { title: 'Scan', detail: 'use grep to find common bug patterns' },
    { title: 'Review', detail: 'manually review findings' }
  ]
}

phase('Scan')
const scanResults = await agent(`
  Use grep to search for potential bug patterns in the codebase.
  Look for:
  - null pointer checks
  - uninitialized variables
  - common error handling patterns
`, {schema: { bugs: [{ pattern: "string", file: "string" }]}})

phase('Review')
const reviewResults = await parallel(scanResults.bugs.map(bug => () =>
  agent(`Manually review the following code snippet:
  File: ${bug.file}
  Pattern: ${bug.pattern}

  Is this a valid bug?
  `, {schema: { isValidBug: "boolean", description: "string" }})))

const confirmedBugs = reviewResults.filter(review => review.isValidBug)
return confirmedBugs;