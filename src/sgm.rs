use std::hash::Hash;
use std::fmt::Debug;
use std::collections::HashMap;

use ordered_float::OrderedFloat;
use petgraph::stable_graph::{StableGraph, NodeIndex};
use petgraph::dot::Dot;

use crate::envs::TensorConvertible;
use crate::replay_buffer::ReplayBuffer;


pub fn dot<S: Debug>(graph: &StableGraph<S, OrderedFloat<f64>>) -> String {
    format!("{:?}", Dot::new(graph)).to_string()
}

impl ReplayBuffer {
    pub fn construct_sgm<S>(
        &self,
        d: fn(&S, &S) -> f64,
        maxdist: f64,
        tau: f64,
    ) -> (StableGraph<S, OrderedFloat<f64>>, HashMap<S, NodeIndex>)
    where
        S: Clone + Eq + Hash + TensorConvertible
    {
        let maxdist = OrderedFloat(maxdist);
        let tau = OrderedFloat(tau);

        // compute the sparse graph
        let mut graph: StableGraph<S, OrderedFloat<f64>> = StableGraph::new();
        let mut indices: HashMap<S, NodeIndex> = HashMap::new();

        // iterate over nodes in the dense graph
        for s1 in self.all_states::<S>() {
            // always accept the first node
            if graph.node_count() == 0 {
                let i1 = graph.add_node(s1.clone());
                indices.insert(s1, i1);

            } else {
                // check if new node is TWC consistent
                let is_twc_consistent =
                    graph
                    .node_weights()
                    .all(|s2| Self::TWC(&s1, s2, tau, d, &graph));

                // info!(
                //     concat!(
                //         "\nTWC consistency of {state:#} with respect to ",
                //         "the graph so far: {consistent:#}.",
                //         "\nThe graph currently has {n_nodes:#} Nodes and ",
                //         "{n_edges:#} Edges.",
                //     ),
                //     state = s1,
                //     consistent = is_twc_consistent,
                //     n_nodes = graph.node_count(),
                //     n_edges = graph.edge_count(),
                // );

                if is_twc_consistent {
                    // add node
                    let i1 = graph.add_node(s1.clone());
                    indices.insert(s1.clone(), i1);


                    // add edges
                    let mut edges_to_add = Vec::new();
                    for s2 in graph.node_weights() {
                        // no self edges
                        if &s1 == s2 {continue;}

                        let d_out = OrderedFloat(d(&s1, s2));
                        let d_in = OrderedFloat(d(s2, &s1));

                        // info!(
                        //     concat!(
                        //         "\nmaxdist = {dist}, ",
                        //         "(d_out: {d_out}, d_in: {d_in})",
                        //     ),
                        //     dist = maxdist,
                        //     d_out = d_out,
                        //     d_in = d_in,
                        // );

                        if d_out < maxdist && d_in < maxdist {
                            edges_to_add.push((i1, indices[s2], d(&s1, s2)));
                            edges_to_add.push((indices[s2], i1, d(s2, &s1)));
                        }
                    }
                    for (a, b, weight) in edges_to_add {
                        graph.add_edge(a, b, OrderedFloat(weight));
                    }
                }
            }
        }

        // info!("\nDotviz: {}", Dot::new(&graph));

        (graph, indices)
    }

    /// (insert equation (3) from SGM here
    ///
    /// TWC(s1, s2, tau, d) is defined as C_in >= tau AND C_out >= tau
    /// given the distance function d
    ///
    /// C_out can be seen as a tau-approximate, Q-irrelevant abstraction.
    /// this means that as we vary tau, we vary the sparsification of the graph
    /// in a way that is irrelevant for the Q-function.
    ///
    /// the (goal-conditioned) Q-function is seen as approximately equivalent
    /// to the true distance function, this implies that distance is interpreted
    /// as expected reward.
    ///
    /// Q(state_s1, action_a, goal_g) ~= d(state_s1, state_s2),
    /// where state_s2 is the results of taking action_a in state_s1
    #[allow(non_snake_case)]
    // #[instrument(skip(graph))]
    fn TWC<S: Eq + Hash>(
        s1: &S,
        s2: &S,
        tau: OrderedFloat<f64>,
        d: fn(&S, &S) -> f64,
        graph: &StableGraph<S, OrderedFloat<f64>>,
    ) -> bool{
        let c_in = Self::c_in(s1, s2, d, graph);
        let c_out = Self::c_out(s1, s2, d, graph);

        // info!("\nC_out is {c_out:#} and C_in is {c_in:#}");

        c_out >= tau && c_in >= tau
    }

    fn c_out<S: Eq + Hash>(
        s1: &S,
        s2: &S,
        d: fn(&S, &S) -> f64,
        graph: &StableGraph<S, OrderedFloat<f64>>,
    ) -> OrderedFloat<f64> {
        graph
            .node_weights()
            .map(|w| OrderedFloat((d(s1, w) - d(s2, w)).abs()))
            .max()
            .expect("StableGraph cannot be empty because we always accept the first node")
    }

    fn c_in<S: Eq + Hash>(
        s1: &S,
        s2: &S,
        d: fn(&S, &S) -> f64,
        graph: &StableGraph<S, OrderedFloat<f64>>,
    ) -> OrderedFloat<f64> {
        graph
            .node_weights()
            .map(|w| OrderedFloat((d(w, s1) - d(w, s2)).abs()))
            .max()
            .expect("StableGraph cannot be empty because we always accept the first node")
    }
}
