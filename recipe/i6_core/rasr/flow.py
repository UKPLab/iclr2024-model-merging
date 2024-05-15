__all__ = [
    "NodeMapping",
    "FlowNetwork",
    "NamedFlowAttribute",
    "FlagDependentFlowAttribute",
    "PathWithPrefixFlowAttribute",
]

import collections
import copy
import itertools as it
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from sisyphus.tools import extract_paths

from .config import RasrConfig


class NodeMapping:
    def __init__(self, mapping):
        """
        :param dict mapping:
        """
        self.mapping = mapping

    def __getitem__(self, key):
        try:
            return self.mapping[key]
        except KeyError:
            if type(key) == str and ":" in key:
                n, p = key.split(":")
                return "%s:%s" % (self[n], p)
            else:
                raise


class FlowNetwork:
    default_flags = {}

    def __init__(self, name="network"):
        """
        :param str name:
        """
        self.name = name
        self.nodes = {}
        self.links = []  # type: list[(str,str)]  # list of (from_name, to_name)
        self.inputs = set()
        self.outputs = set()
        self.params = set()

        self.named_attributes = {}
        self.hidden_inputs = set()

        self.flags = self.default_flags.copy()

        self.config = None
        self.post_config = None

    def unique_name(self, name):
        """
        :param str name:
        :rtype: str
        """
        orig_name = name
        i = 1
        while name in self.nodes:
            name = "%s~%d" % (orig_name, i)
            i += 1
        return name

    def add_node(self, filter, name, attr=None, **kwargs):
        if attr is None:
            attr = {}

        name = self.unique_name(name)
        attributes = {}
        attributes.update(attr)
        attributes.update(**kwargs)
        attributes["filter"] = filter

        self.nodes[name] = attributes

        # search for NamedFlowAttribute
        for v in attributes.values():
            if isinstance(v, NamedFlowAttribute):
                self.named_attributes[v.name] = v

        return name

    def remove_node(self, name):
        del self.nodes[name]
        f = lambda ln: ln.split(":")[0] != name
        self.links = [l for l in self.links if f(l[0]) and f(l[1])]

    def link(self, from_name, to_name):
        """
        :param str from_name:
        :param str to_name:
        """
        if (from_name, to_name) not in self.links:
            self.links.append((from_name, to_name))

    def unlink(self, from_name=None, to_name=None):
        self.links = [
            l
            for l in self.links
            if not (
                (from_name is None or l[0] == from_name)
                and (to_name is None or l[1] == to_name)
            )
        ]

    def add_input(self, name):
        self.inputs = _smart_union(self.inputs, name)

    def add_output(self, name):
        self.outputs = _smart_union(self.outputs, name)

    def add_param(self, name):
        self.params = _smart_union(self.params, name)

    def add_flags(self, flags):
        self.flags.update(flags)

    def add_net(self, net):
        assert isinstance(net, FlowNetwork)
        self.add_param(net.params)
        self.add_flags(net.flags)
        mapping = {}
        for name, node in net.nodes.items():
            mapping[name] = self.add_node(name=name, **node)
        mapping = NodeMapping(mapping)
        for from_name, to_name in net.get_internal_links():
            self.link(mapping[from_name], mapping[to_name])
        self.add_hidden_input(net.hidden_inputs)

        for original_config, attr in [
            (net.config, "config"),
            (net.post_config, "post_config"),
        ]:
            if original_config is None:
                continue

            if getattr(self, attr) is None:
                setattr(self, attr, RasrConfig())

            self_config = getattr(self, attr)

            for k, v in original_config._items():
                self_config[mapping[k]]._update(v)

        return mapping

    def add_hidden_input(self, input):
        """in case a Path has to be converted to a string that is then added to the network"""
        self.hidden_inputs.update(extract_paths(input))

    def interconnect(self, a, node_mapping_a, b, node_mapping_b, mapping=None):
        """assuming a and b are FlowNetworks that have already been added to this net,
        the outputs of a are linked to the inputs of b,
        optionally a mapping between the ports can be specified
        """
        if mapping is None:
            ordered_mapping = collections.OrderedDict(
                (p, p)
                for p in sorted(
                    set(a.get_output_ports()).intersection(set(b.get_input_ports()))
                )
            )
        else:
            ordered_mapping = collections.OrderedDict(
                (k, mapping[k]) for k in sorted(mapping.keys())
            )
        for src_port, dst_port in ordered_mapping.items():
            for from_name, to_name in it.product(
                a.get_output_links(src_port), b.get_input_links(dst_port)
            ):
                self.link(node_mapping_a[from_name], node_mapping_b[to_name])

    def interconnect_inputs(self, net, node_mapping, mapping=None):
        """assuming net has been added to self,
        link all of self's inputs to net's inputs,
        optionally a mapping between the ports can be specified
        """
        if mapping is None:
            ordered_mapping = collections.OrderedDict(
                (p, p) for p in sorted(net.get_input_ports())
            )
        else:
            ordered_mapping = collections.OrderedDict(
                (k, mapping[k]) for k in sorted(mapping.keys())
            )
        for original_port, new_port in ordered_mapping.items():
            self.add_input(new_port)
            for dst in net.get_input_links(original_port):
                self.link("%s:%s" % (self.name, new_port), node_mapping[dst])

    def interconnect_outputs(self, net, node_mapping, mapping=None):
        """assuming net has been added to self,
        link all of net's outputs to self's outputs,
        optionally a mapping between the ports can be specified
        """
        if mapping is None:
            ordered_mapping = collections.OrderedDict(
                (p, p) for p in sorted(net.get_output_ports())
            )
        else:
            ordered_mapping = collections.OrderedDict(
                (k, mapping[k]) for k in sorted(mapping.keys())
            )
        for original_port, new_port in ordered_mapping.items():
            self.add_output(new_port)
            for src in net.get_output_links(original_port):
                self.link(node_mapping[src], "%s:%s" % (self.name, new_port))

    def subnet_from_node(self, node_name):
        """creates a new net where only nodes that follow the given
        node are retained. nodes before the specified node are
        not included. links between one retained node and one
        not retained one are returned aswell.
        this function is usefull if a part of a net should be
        duplicated without copying the other part
        """
        assert node_name in self.nodes

        net = FlowNetwork(name=self.name)
        net.add_input(self.get_input_ports())
        net.add_output(self.get_output_ports())

        nodes = [node_name]
        processed_nodes = set()
        mapping = {}
        internal_links = self.get_internal_links()

        # add all nodes that follow the given node
        while len(nodes) > 0:
            name = nodes.pop()
            processed_nodes.add(name)

            attr = copy.deepcopy(self.nodes[name])
            mapping[name] = net.add_node(attr["filter"], name, attr)

            subsequent_nodes = set()

            for l in internal_links:
                if l[0] == name or l[0].startswith(name + ":"):
                    subsequent_nodes.add(l[1].split(":")[0])

            nodes.extend(sorted(subsequent_nodes.difference(processed_nodes)))

        broken_links = []

        # now add all the links
        for l in internal_links:
            name_in = l[0].split(":")[0]
            name_out = l[1].split(":")[0]
            if name_in in processed_nodes:
                net.link(mapping[l[0]], mapping[l[1]])
            elif name_out in processed_nodes:
                broken_links.append((l[0], mapping[l[1]]))

        for port in self.get_input_ports():
            for l in self.get_input_links(port):
                if l.split(":")[0] in processed_nodes:
                    net.link("%s:%s" % (net.name, port), mapping[l])

        for port in self.get_output_ports():
            for l in self.get_output_links(port):
                if l.split(":")[0] in processed_nodes:
                    net.link(mapping[l], "%s:%s" % (net.name, port))

        return net, broken_links

    def write_to_file(self, file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(repr(self))

    def get_internal_links(self):
        return list(
            l
            for l in self.links
            if not l[0].startswith(self.name + ":")
            and not l[1].startswith(self.name + ":")
        )

    def get_input_links(self, input_port):
        return list(
            l[1] for l in self.links if l[0] == "%s:%s" % (self.name, input_port)
        )

    def get_output_links(self, output_port):
        """
        :param str output_port:
        :return: list of from_name
        :rtype: list[str]
        """
        return list(
            l[0] for l in self.links if l[1] == "%s:%s" % (self.name, output_port)
        )

    def get_input_ports(self):
        return sorted(self.inputs)

    def get_output_ports(self):
        return sorted(self.outputs)

    def get_node_names_by_filter(self, filter_name):
        return [k for k, v in self.nodes.items() if v["filter"] == filter_name]

    def contains_filter(self, filter_name):
        return any(n.get("filter", None) == filter_name for n in self.nodes.values())

    def apply_config(self, path, config, post_config):
        config[path]._update(self.config)
        post_config[path]._update(self.post_config)

    def __compute_node_order(self):
        dependencies = collections.defaultdict(set)
        result = []
        added_nodes = set()
        missing_nodes = set(self.nodes.keys())

        for link in self.links:
            from_node = link[0].split(":")[0]
            to_node = link[1].split(":")[0]

            if from_node != self.name and to_node != self.name:
                dependencies[to_node].add(from_node)

        while len(missing_nodes) > 0:
            for n in missing_nodes:
                if all(dep in added_nodes for dep in dependencies[n]):
                    result.append(n)
                    added_nodes.add(n)
                    missing_nodes.remove(n)
                    break
            else:
                # no node added => contains loops => add one node regardless of dependencies
                n = missing_nodes.pop()
                result.append(n)
                added_nodes.add(n)

        return result

    def __repr__(self):
        root = ET.Element("network", name=self.name)
        links_by_target_node = collections.defaultdict(list)
        for l in self.links:
            links_by_target_node[l[1].split(":")[0]].append(l)

        for node, names in zip(
            ["in", "out", "param"], [self.inputs, self.outputs, self.params]
        ):
            for n in sorted(names):
                ET.SubElement(root, node, name=n)

        for name in self.__compute_node_order():
            attr = self.nodes[name]
            nstr = {"name": name}
            for k, v in attr.items():
                while isinstance(v, FlowAttribute):
                    v = v.get(self)
                if type(v) == bool:
                    nstr[k] = str(v).lower()
                else:
                    nstr[k] = str(v)

            ET.SubElement(root, "node", nstr)

            if name in links_by_target_node:
                for link in links_by_target_node[name]:
                    ET.SubElement(root, "link", {"from": link[0], "to": link[1]})
                del links_by_target_node[name]

        for links in links_by_target_node.values():
            for link in links:
                ET.SubElement(root, "link", {"from": link[0], "to": link[1]})

        dom = minidom.parseString(ET.tostring(root, "utf-8"))
        return dom.toprettyxml(indent="  ")

    def __sis_state__(self):
        def get_val(a):
            return get_val(a.get(self)) if isinstance(a, FlowAttribute) else a

        nodes = {
            name: {k: get_val(v) for k, v in attr.items()}
            for name, attr in self.nodes.items()
        }
        state = {
            "name": self.name,
            "nodes": nodes,
            "links": self.links,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
            "hidden_inputs": self.hidden_inputs,
            "config": self.config,
            "post_config": self.post_config,
        }
        return state


class FlowAttribute:
    def get(self, net):
        pass


class NamedFlowAttribute(FlowAttribute):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get(self, net):
        return self.value


class FlagDependentFlowAttribute(FlowAttribute):
    def __init__(self, flag, alternatives):
        self.flag = flag
        self.alternatives = alternatives

    def get(self, net):
        return self.alternatives[net.flags[self.flag]]


class PathWithPrefixFlowAttribute(FlowAttribute):
    def __init__(self, prefix, path):
        self.prefix = prefix
        self.path = path

    def get(self, net):
        return "%s:%s" % (self.prefix, self.path)


def _smart_union(s, e):
    if type(e) == list or type(e) == set:
        return s.union(e)
    return s.union([e])
