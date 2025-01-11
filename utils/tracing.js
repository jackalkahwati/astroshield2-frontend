const opentelemetry = require('@opentelemetry/api');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { SimpleSpanProcessor } = require('@opentelemetry/sdk-trace-base');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const crypto = require('crypto');

class Tracing {
  constructor() {
    this.provider = null;
    this.tracer = null;
  }

  initialize() {
    // Configure the trace provider
    this.provider = new NodeTracerProvider({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: 'astroshield-service',
      }),
    });

    // Configure the Jaeger exporter
    const exporter = new JaegerExporter({
      endpoint: process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
    });

    // Add the SpanProcessor to the provider
    this.provider.addSpanProcessor(new SimpleSpanProcessor(exporter));

    // Register the provider
    this.provider.register();

    // Get a tracer
    this.tracer = opentelemetry.trace.getTracer('astroshield-tracer');
  }

  startTrace(name, options = {}) {
    const span = this.tracer.startSpan(name, options);
    return {
      traceId: crypto.randomBytes(16).toString('hex'),
      spanId: crypto.randomBytes(8).toString('hex'),
      parentId: options.parentId,
      name,
      startTime: Date.now(),
      attributes: {},
      _span: span,
      setAttribute: function(key, value) {
        this.attributes[key] = value;
        this._span.setAttribute(key, value);
      },
      end: function() {
        this._span.end();
      }
    };
  }

  async startChildSpan(name, parentTrace, options = {}) {
    const context = opentelemetry.trace.setSpan(
      opentelemetry.context.active(),
      parentTrace._span
    );

    const childSpan = this.tracer.startSpan(name, options, context);
    return {
      traceId: parentTrace.traceId,
      spanId: crypto.randomBytes(8).toString('hex'),
      parentId: parentTrace.spanId,
      name,
      startTime: Date.now(),
      attributes: {},
      _span: childSpan,
      setAttribute: function(key, value) {
        this.attributes[key] = value;
        this._span.setAttribute(key, value);
      },
      end: function() {
        this._span.end();
      }
    };
  }

  getCurrentSpan() {
    return opentelemetry.trace.getSpan(opentelemetry.context.active());
  }

  withSpan(name, fn, options = {}) {
    const span = this.startSpan(name, options);
    return opentelemetry.context.with(
      opentelemetry.trace.setSpan(opentelemetry.context.active(), span._span),
      async () => {
        try {
          const result = await fn(span);
          span.end();
          return result;
        } catch (error) {
          span._span.recordException(error);
          span._span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
          span.end();
          throw error;
        }
      }
    );
  }

  addAttribute(key, value) {
    const span = this.getCurrentSpan();
    if (span) {
      span.setAttribute(key, value);
    }
  }

  addEvent(name, attributes = {}) {
    const span = this.getCurrentSpan();
    if (span) {
      span.addEvent(name, attributes);
    }
  }

  setError(error) {
    const span = this.getCurrentSpan();
    if (span) {
      span.recordException(error);
      span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
    }
  }
}

module.exports = new Tracing(); 