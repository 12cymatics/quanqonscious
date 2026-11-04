# Sites and interface instructions

Status: project overlay for implementation and review. It neither publishes a site nor grants deployment authority.

## Truthful interface contract

Use the shared mathematical-foundation and integrity rules for the calculations behind the interface. Geometry may also organise the visual design, but layout ratios and symbolic motifs must not be described as scientific validation. Prioritise readable controls and correct operation over ornamental complexity.

For each view, state its purpose, input source, active model, units, constraints and output meaning. Distinguish measured data, computed results, supplied data, authored examples and hypotheses. Do not fabricate live values, customer evidence, test results or scientific authority. Label sample data and keep it separate from actual results.

Every visible control must perform its stated function or be clearly unavailable with a reason. Do not ship decorative toggles, fabricated progress, solver replacements or success messages emitted before verification. Missing capability, input or permission is a visible state, not a reason to substitute a fake output.

## Interaction and accessibility

- Group related inputs with clear units, meaningful presets and manual adjustment where required. A preset must expose its actual values and model assumptions.
- Preserve inputs during recoverable navigation and make reset/destructive actions explicit. Verify changes are applied to the intended object or run.
- Show pending, running, paused, completed, failed, cancelled and unsupported states only when the underlying process supports them. Explain whether pause stops display, solver or both.
- Use semantic controls, keyboard navigation, visible focus, readable contrast, accessible names and legible responsive layouts. Do not convey status by colour alone.
- Check empty, invalid, extreme, loading and failure states. Test repeated clicks, interruption, Back/Forward, reload and narrow viewports, not just a polished default screenshot.
- Preserve existing architecture, user content and asset licensing. Avoid unrequested analytics, tracking, external services or permission expansion.

## Integration and release

Identify the actual project, source revision and environment before edits. Keep the interface connected to the real implementation contract, including its precision, capability and error limits. Do not change mathematical scope to fit the UI. Preserve all specified equations and their formulation without approximations, reductions, truncations, transformations or replacements.

Test the implemented app in the supported browser workflow with real interactions and console/network checks. Verify workers, assets, downloads and exported files. Separate simulated test inputs from real integrations. If only mocks or static inspection were possible, say so.

After an authorized publish, retain the platform-native deployment acknowledgement, returned URL and available revision/build identity. This can establish deployment status, not feature correctness. Perform relevant browser or integration tests when claiming runtime behaviour, persistence or target-device support; do not require a redundant browser visit merely to restate a trustworthy hosting acknowledgement. A source upload is not automatically a deployment, and local tests do not prove live behaviour. Report deployment and functional coverage separately.
