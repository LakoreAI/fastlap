pub fn supported_algorithms() -> Vec<&'static str> {
    vec![
        "lapjv",
        "hungarian",
        "lapmod",
        "subgradient",
        "auction",
        "dantzig",
    ]
}
