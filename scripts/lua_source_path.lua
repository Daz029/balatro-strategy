--- Resolve the read-only Balatro 1.0.1o source tree used by Lua tooling.
---
--- Resolution order is shared with tests/_lua_source.py:
---   1. BALATRO_SOURCE
---   2. <project_root>/balatro_source (the historical checkout layout)
---   3. ~/Code/Code/balatro-strategy/balatro_source/Balatro

local M = {}

local function trim_trailing_separator(path)
    return (path:gsub("[/\\]+$", ""))
end

local function source_tree_exists(path)
    local handle = io.open(trim_trailing_separator(path) .. "/card.lua", "r")
    if handle then
        handle:close()
        return true
    end
    return false
end

function M.resolve(project_root)
    local configured = os.getenv("BALATRO_SOURCE")
    if configured and configured ~= "" then
        return trim_trailing_separator(configured)
    end

    project_root = project_root or "./"
    local checkout_source = trim_trailing_separator(project_root) .. "/balatro_source"
    if source_tree_exists(checkout_source) then
        return checkout_source
    end

    local home = os.getenv("HOME") or os.getenv("USERPROFILE") or "~"
    return trim_trailing_separator(home) .. "/Code/Code/balatro-strategy/balatro_source/Balatro"
end

return M
