pub fn supported_algorithms() -> &'static [&'static str] {
    &[
        "lapjv",
        "hungarian",
        "lapmod",
        "subgradient",
        "auction",
        "dantzig",
    ]
}
