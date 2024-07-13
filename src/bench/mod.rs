#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use petgraph::graph::{DefaultIx, EdgeIndex, NodeIndex};
    use std::collections::HashMap;
    use std::fs::read_to_string;
    use std::time::Instant;
    use sucds::bit_vectors::Rank9Sel;

    // Tinydg = "src/bench/input/tinyDG.txt",
    // Mediumdg = "src/bench/input/mediumDG.txt",
    // Largedg = "src/bench/input/largeDG.txt",
    // Tinyewd = "src/bench/input/tinyEWD.txt",
    // Mediumewd = "src/bench/input/mediumEWD.txt",
    // Largeewd = "src/bench/input/largeEWD.txt",

    fn parse_graph(
        filename: &str,
        weighted: bool,
    ) -> (usize, usize, (Vec<(usize, usize)>, Vec<f64>)) {
        let lines_as_vec = read_lines(filename);
        let number_of_nodes = lines_as_vec[0].parse::<usize>().unwrap();
        let number_of_edges = lines_as_vec[1].parse::<usize>().unwrap();
        let temp_edges_vec = &lines_as_vec[2..];
        if weighted {
            (
                number_of_nodes,
                number_of_edges,
                parse_weighted(temp_edges_vec, number_of_edges),
            )
        } else {
            (
                number_of_nodes,
                number_of_edges,
                parse_non_weighted(temp_edges_vec, number_of_edges),
            )
        }
    }

    fn parse_weighted(
        temp_edges_vec: &[String],
        number_of_edges: usize,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for i in 0..number_of_edges {
            let line_iter = temp_edges_vec[i].split(" ");
            let line = line_iter.filter(|s| s.len() != 0).collect::<Vec<&str>>();

            edges.push((
                line[0].parse::<usize>().unwrap(),
                line[1].parse::<usize>().unwrap(),
            ));
            weights.push(line[2].parse::<f64>().unwrap());
        }
        (edges, weights)
    }

    fn parse_non_weighted(
        temp_edges_vec: &[String],
        number_of_edges: usize,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let weights = vec![1.0; number_of_edges];
        for i in 0..number_of_edges {
            let line_iter = temp_edges_vec[i].split(" ");
            let line = line_iter.filter(|s| s.len() != 0).collect::<Vec<&str>>();

            edges.push((
                line[0].parse::<usize>().unwrap(),
                line[1].parse::<usize>().unwrap(),
            ));
        }
        (edges, weights)
    }

    fn read_lines(filename: &str) -> Vec<String> {
        read_to_string(filename)
            .unwrap()
            .lines()
            .map(String::from)
            .collect()
    }

    fn create_directed_benchmark_graph(
        filename: &str,
        has_weighted_edges: bool,
    ) -> GraphModel<usize, f64, f64, petgraph::Directed> {
        match has_weighted_edges {
            true => create_directed_weighted_benchmark_graph(filename),
            false => create_directed_unweighted_benchmark_graph(filename),
        }
    }

    fn create_undirected_benchmark_graph(
        filename: &str,
        has_weighted_edges: bool,
    ) -> GraphModel<usize, f64, f64, petgraph::Undirected> {
        match has_weighted_edges {
            true => create_undirected_weighted_benchmark_graph(filename),
            false => create_undirected_unweighted_benchmark_graph(filename),
        }
    }

    fn create_directed_weighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Directed> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let parsed_file = parse_graph(filename, true);
        // creates Nodes from 0 to number_of_nodes, which is the first line of the
        // passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the second line of the
        // passed file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    fn create_undirected_weighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Undirected> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let parsed_file = parse_graph(filename, true);
        // creates Nodes from 0 to number_of_nodes, which is the first line of the
        // passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the second line of the passed
        // file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    // unweighted being all edges weights are set to 1.0
    fn create_directed_unweighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Directed> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let parsed_file = parse_graph(filename, false);
        // creates Nodes from 0 to number_of_nodes, which is the first line fo the passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the second line of the passed file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    // unweighted being all edge weights are set to 1.0
    fn create_undirected_unweighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Undirected> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let parsed_file = parse_graph(filename, false);
        // creates Nodes from 0 to number_of_nodes, which is the first line of the passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the seconds line of the passed file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    fn bench_directed_create(filename: &str) {
        let graph = create_directed_unweighted_benchmark_graph(filename);
        let now = Instant::now();
        if let Ok(_) = WaveModel::try_from(graph) {
            println!("Directed_create: {:.3?}", now.elapsed());
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_undirected_create(filename: &str) {
        let graph = create_undirected_unweighted_benchmark_graph(filename);
        let now = Instant::now();
        if let Ok(_) = WaveModel::try_from(graph) {
            println!("Undirected_create: {:.3?}", now.elapsed());
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_directed_create_add_node(filename: &str) {
        let graph = create_directed_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let now = Instant::now();
            model.add_node(usize::MAX, 1.0);
            println!("Directed_add_node: {:.3?}", now.elapsed());
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_directed_create_add_edge(filename: &str) {
        let graph = create_directed_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let v1 = model.add_node(usize::MAX, 1.0);
            let v2 = model.add_node(usize::MAX - 1, 1.0);
            let now = Instant::now();
            let _ = model.add_edge(usize::MAX, 1.0, v1, v2);
            println!("Directed_add_edge: {:.3?}", now.elapsed());
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_undirected_create_add_node(filename: &str) {
        let graph = create_undirected_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let now = Instant::now();
            model.add_node(usize::MAX, 1.0);
            println!("Undirected_add_node: {:.3?}", now.elapsed());
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_undirected_create_add_edge(filename: &str) {
        let graph = create_undirected_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let v1 = model.add_node(usize::MAX, 1.0);
            let v2 = model.add_node(usize::MAX - 1, 1.0);
            let now = Instant::now();
            let _ = model.add_edge(usize::MAX, 1.0, v1, v2);
            println!("Undirected_add_edge: {:.3?}", now.elapsed());
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_directed_create_remove_node(filename: &str, iterations: u8) {
        let graph = create_directed_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let now = Instant::now();
            for _ in 0..iterations {
                let _ = model.remove_node(&(0));
            }
            println!(
                "Directed_remove_node_{:?}: {:.3?}",
                iterations,
                now.elapsed()
            );
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_directed_create_remove_edge(filename: &str, iterations: u8) {
        let graph = create_directed_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let now = Instant::now();
            for _ in 0..iterations {
                let _ = model.remove_edge(&(0));
            }
            println!(
                "Directed_remove_edge_{:?}: {:.3?}",
                iterations,
                now.elapsed()
            );
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_undirected_create_remove_node(filename: &str, iterations: u8) {
        let graph = create_undirected_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let now = Instant::now();
            for _ in 0..iterations {
                let _ = model.remove_node(&(0));
            }
            println!(
                "Undirected_remove_node_{:?}: {:.3?}",
                iterations,
                now.elapsed()
            );
            assert!(true);
        } else {
            assert!(false);
        }
    }

    fn bench_undirected_create_remove_edge(filename: &str, iterations: u8) {
        let graph = create_undirected_unweighted_benchmark_graph(filename);
        if let Ok(mut model) = WaveModel::try_from(graph) {
            let now = Instant::now();
            for _ in 0..iterations {
                let _ = model.remove_edge(&(0));
            }
            println!(
                "Undirected_remove_edge_{:?}: {:.3?}",
                iterations,
                now.elapsed()
            );
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    #[ignore]
    pub fn exec_bench() {
        // // TinyDG
        // bench_directed_create("src/bench/input/tinyDG.txt");
        // bench_directed_create_add_node("src/bench/input/tinyDG.txt");
        // bench_directed_create_add_edge("src/bench/input/tinyDG.txt");
        // bench_directed_create_remove_edge("src/bench/input/tinyDG.txt", 1 as u8);
        // bench_directed_create_remove_node("src/bench/input/tinyDG.txt", 1 as u8);

        // bench_undirected_create("src/bench/input/tinyDG.txt");
        // bench_undirected_create_add_node("src/bench/input/tinyDG.txt");
        // bench_undirected_create_add_edge("src/bench/input/tinyDG.txt");
        // bench_undirected_create_remove_edge("src/bench/input/tinyDG.txt", 1 as u8);
        // bench_undirected_create_remove_node("src/bench/input/tinyDG.txt", 1 as u8);

        // // MediumDG
        // bench_directed_create("src/bench/input/mediumDG.txt");
        // bench_directed_create_add_node("src/bench/input/mediumDG.txt");
        // bench_directed_create_add_edge("src/bench/input/mediumDG.txt");
        // bench_directed_create_remove_edge("src/bench/input/mediumDG.txt", 1 as u8);
        // bench_directed_create_remove_node("src/bench/input/mediumDG.txt", 1 as u8);

        // bench_undirected_create("src/bench/input/mediumDG.txt");
        // bench_undirected_create_add_node("src/bench/input/mediumDG.txt");
        // bench_undirected_create_add_edge("src/bench/input/mediumDG.txt");
        // bench_undirected_create_remove_edge("src/bench/input/mediumDG.txt", 1 as u8);
        // bench_undirected_create_remove_node("src/bench/input/mediumDG.txt", 1 as u8);

        // // LargeDG
        // bench_directed_create("src/bench/input/largeDG.txt");
        // bench_directed_create_add_node("src/bench/input/largeDG.txt");
        // bench_directed_create_add_edge("src/bench/input/largeDG.txt");
        // bench_directed_create_remove_edge("src/bench/input/largeDG.txt", 1 as u8);
        // bench_directed_create_remove_node("src/bench/input/largeDG.txt", 1 as u8);

        // bench_undirected_create("src/bench/input/largeDG.txt");
        // bench_undirected_create_add_node("src/bench/input/largeDG.txt");
        // bench_undirected_create_add_edge("src/bench/input/largeDG.txt");
        // bench_undirected_create_remove_edge("src/bench/input/largeDG.txt", 1 as u8);
        // bench_undirected_create_remove_node("src/bench/input/largeDG.txt", 1 as u8);
    }

    #[test]
    #[ignore]
    pub fn bench_custom_timing() {
        // Change if custom timing is wanted
        if false {
            assert!(true);
            return;
        }

        // tiny
        bench_directed_create("src/bench/input/tinyDG.txt");
        bench_directed_create_add_node("src/bench/input/tinyDG.txt");
        bench_directed_create_add_edge("src/bench/input/tinyDG.txt");
        bench_directed_create_remove_edge("src/bench/input/tinyDG.txt", 1 as u8);
        bench_directed_create_remove_node("src/bench/input/tinyDG.txt", 1 as u8);

        bench_undirected_create("src/bench/input/tinyDG.txt");
        bench_undirected_create_add_node("src/bench/input/tinyDG.txt");
        bench_undirected_create_add_edge("src/bench/input/tinyDG.txt");
        bench_undirected_create_remove_edge("src/bench/input/tinyDG.txt", 1 as u8);
        bench_undirected_create_remove_node("src/bench/input/tinyDG.txt", 1 as u8);

        // medium
        bench_directed_create("src/bench/input/mediumDG.txt");
        bench_directed_create_add_node("src/bench/input/mediumDG.txt");
        bench_directed_create_add_edge("src/bench/input/mediumDG.txt");
        bench_directed_create_remove_edge("src/bench/input/mediumDG.txt", 1 as u8);
        bench_directed_create_remove_node("src/bench/input/mediumDG.txt", 1 as u8);

        bench_undirected_create("src/bench/input/mediumDG.txt");
        bench_undirected_create_add_node("src/bench/input/mediumDG.txt");
        bench_undirected_create_add_edge("src/bench/input/mediumDG.txt");
        bench_undirected_create_remove_edge("src/bench/input/mediumDG.txt", 1 as u8);
        bench_undirected_create_remove_node("src/bench/input/mediumDG.txt", 1 as u8);

        // large
        bench_directed_create("src/bench/input/largeDG.txt");
        bench_directed_create_add_node("src/bench/input/largeDG.txt");
        bench_directed_create_add_edge("src/bench/input/largeDG.txt");
        bench_directed_create_remove_edge("src/bench/input/largeDG.txt", 1 as u8);
        bench_directed_create_remove_node("src/bench/input/largeDG.txt", 1 as u8);
        bench_directed_create_remove_edge("src/bench/input/largeDG.txt", 10 as u8);
        bench_directed_create_remove_edge("src/bench/input/largeDG.txt", 10 as u8);

        bench_undirected_create("src/bench/input/largeDG.txt");
        bench_undirected_create_add_node("src/bench/input/largeDG.txt");
        bench_undirected_create_add_edge("src/bench/input/largeDG.txt");
        bench_undirected_create_remove_edge("src/bench/input/largeDG.txt", 1 as u8);
        bench_undirected_create_remove_node("src/bench/input/largeDG.txt", 1 as u8);
        bench_undirected_create_remove_edge("src/bench/input/largeDG.txt", 10 as u8);
        bench_undirected_create_remove_edge("src/bench/input/largeDG.txt", 10 as u8);
    }
}
