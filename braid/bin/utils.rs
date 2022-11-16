fn dashed_line(cell_width: usize, text_width: usize) -> String {
    let n_spaces = cell_width - text_width;
    " ".repeat(n_spaces) + "─".repeat(text_width).as_str()
}

pub fn print_table(header: Vec<String>, rows: Vec<Vec<String>>) {
    // XXX: tables are built assuming all chars in are from the usual English
    // charset. Weird unicode things will not give the desired behavior with
    // the .len() method.
    let ncols = header.len();
    rows.iter().enumerate().for_each(|(rowix, row)| {
        if row.len() != ncols {
            panic!(
                "There are {} columns in the header, but row {} has {} entries",
                ncols,
                rowix,
                row.len()
            )
        }
    });

    let mut widths: Vec<_> = header.iter().map(|entry| entry.len()).collect();

    rows.iter().for_each(|row| {
        row.iter().enumerate().for_each(|(colix, entry)| {
            let width = entry.len();
            if width > widths[colix] {
                widths[colix] = width;
            }
        });
    });

    for (cell, cell_width) in header.iter().zip(widths.iter()) {
        print!("  ");
        print!("{}", " ".repeat(cell_width - cell.len()));
        print!("{}", cell);
    }
    println!();

    for (cell, cell_width) in header.iter().zip(widths.iter()) {
        print!("  ");
        print!("{}", dashed_line(*cell_width, cell.len()));
    }
    println!();

    for row in rows.iter() {
        for (cell, cell_width) in row.iter().zip(widths.iter()) {
            print!("  ");
            print!("{}", " ".repeat(cell_width - cell.len()));
            print!("{}", cell);
        }
        println!();
    }
}
